import os
import math
import copy
import argparse
import matplotlib.pyplot as plt
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import mmcv
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPoint, box
from mmdet3d.datasets import NuScenesDataset
from nuscenes.utils.geometry_utils import view_points
from mmdet3d.core.bbox.box_np_ops import points_cam2img
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

import sys
import warnings
class suppress_output_and_warnings:
    def __enter__(self):
        self._original_stdout = sys.stdout  # 保存原始的stdout
        self._original_stderr = sys.stderr  # 保存原始的stderr
        sys.stdout = open(os.devnull, 'w')  # 重定向stdout到null设备
        sys.stderr = open(os.devnull, 'w')  # 重定向stderr到null设备
        self._original_warning_filters = warnings.filters[:]  # 保存当前的warning过滤器
        warnings.filterwarnings("ignore")  # 暂时忽略所有警告
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # 关闭重定向的文件
        sys.stderr.close()  # 关闭重定向的错误输出文件
        sys.stdout = self._original_stdout  # 恢复原始stdout
        sys.stderr = self._original_stderr  # 恢复原始stderr
        warnings.filters.clear()  # 清空当前的warning过滤器
        warnings.filters.extend(self._original_warning_filters)  # 恢复原来的warning过滤器

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

ego_width, ego_length = 1.85, 4.084

# 他车的质量相对于自车的倍数
MASS = {
    'barrier': 2.5,
    'bicycle': 0.5,
    'bus': 2.0,
    'car': 1.0,
    'construction_vehicle': 1.0,
    'motorcycle': 1.0,
    'pedestrian': 0.5,
    'traffic_cone': 0.0,
    'trailer': 2.5,
    'truck': 2.5
}
# 他车的危险系数
Risk_K = {
    'barrier': 2.5,
    'bicycle': 0.5,
    'bus': 2.0,
    'car': 1.0,
    'construction_vehicle': 1.0,
    'motorcycle': 1.0,
    'pedestrian': 0.5,
    'traffic_cone': 0.0,
    'trailer': 2.5,
    'truck': 2.5
}
# 环境因素
Risk_C = 1.0
# 横向衰减系数
Risk_beta = 0.5


def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
        # 手动减少数据量
        # import random
        # train_scenes = random.sample(train_scenes, 1)
        # val_scenes = random.sample(val_scenes, 1)
        # train_scenes = train_scenes[10:15]
        # val_scenes = val_scenes[10:15]
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
        # 手动减少数据量
        # import random
        # train_scenes = random.sample(train_scenes, 1)
        # train_scenes = train_scenes[10:15]
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    # 找出可用的场景，主要是根据是否有本地数据进行判断
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(len(train_scenes), len(val_scenes)))

    # 为train和val提取数据
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path, '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)

def get_road_type(nusc, map_location, ego_pose):
    # 获取地图
    with suppress_output_and_warnings():
        nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_location)
    # 获取车辆translation
    ego_translation = ego_pose['translation']
    
    road_type = None

    # 获取最近的道路
    closest_lane_token = nusc_map.get_closest_lane(ego_translation[0], ego_translation[1], radius=5.0)
    if closest_lane_token is None:
        # 当前车辆没有最近的道路
        road_type = "free road"
    else:
        # 获取lane_record
        lane_record = nusc_map.get_arcline_path(closest_lane_token)

        # 计算当前lane_record的总长度
        length_total = sum([sum(path['segment_length']) for path in lane_record])
        # 计算自车位置在车道上最近的点
        _, distance_along_lane = arcline_path_utils.project_pose_to_lane(ego_translation[:2], lane_record)
        distance_along_lane = min(distance_along_lane, length_total - 1e-3)
        # 计算投影点处的斜率
        curvature = arcline_path_utils.get_curvature_at_distance_along_lane(distance_along_lane, lane_record)
        if curvature < 0.01:
            road_type = "straight road"
        else:
            road_type = "curved road"

        # 获取road_segment，看看当前是不是路口
        road_segment_token = nusc_map.layers_on_point(*lane_record[0]['start_pose'][:2], layer_names=['road_segment'])['road_segment']
        if len(road_segment_token) > 0:
            road_segment = nusc_map.get('road_segment', road_segment_token)
            road_type = "intersection" if road_segment['is_intersection'] else road_type

    # 将road_type转换为one-hot形式
    types = ['free road', 'intersection','straight road', 'curved road']
    road_type_one_hot = np.array([1 if road_type == t else 0 for t in types])

    return road_type, road_type_one_hot, types

def get_traffic_condition(agents, names):
    def cal_num_in_range(range_lon, range_lat):
        num = 0
        for agent, name in zip(agents_useful, names):
            if range_lon[0] <= agent[0] <= range_lon[1] and range_lat[0] <= agent[1] <= range_lat[1]:
                num += 1
        return num

    # 通过names过滤一遍
    agents_useful = []
    for agent, name in zip(agents, names):
        if name in ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'trailer', 'truck']:
            agents_useful.append(agent)

    if cal_num_in_range([-5, 5], [0, 10]) >= 2:
        traffic_condition = "heavy traffic"
    elif cal_num_in_range([-10, 10], [0, 15]) >= 2:
        traffic_condition = "normal traffic"
    elif cal_num_in_range([-15, 15], [0, 25]) >= 2:
        traffic_condition = "smooth traffic"
    else:
        traffic_condition = "free traffic"

    # 将traffic_condition转换为one-hot形式
    types = ['free traffic', 'heavy traffic', 'normal traffic','smooth traffic']
    traffic_condition_one_hot = np.array([1 if traffic_condition == t else 0 for t in types])

    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # fig = plt.figure()
    # for agent in agents:
    #     plt.plot(agent[0], agent[1], 'go')
    # rect = patches.Rectangle((-5, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    # plt.gca().add_patch(rect)
    # rect = patches.Rectangle((-10, 0), 20, 30, linewidth=1, edgecolor='r', facecolor='none')
    # plt.gca().add_patch(rect)
    # rect = patches.Rectangle((-15, 0), 30, 50, linewidth=1, edgecolor='r', facecolor='none')
    # plt.gca().add_patch(rect)
    # plt.gca().set_aspect('equal')
    # plt.close(fig)

    return traffic_condition, traffic_condition_one_hot, types

def compute_point_risk_value(ego_dxy, agent_dxy, agent_name, distance):
    m = MASS.get(agent_name, 2.0)
    k = Risk_K.get(agent_name, 2.0)
    c = Risk_C
    beta = Risk_beta

    # 位移变化替代速度
    ego_v = np.linalg.norm(ego_dxy)
    agent_v = np.linalg.norm(agent_dxy)

    # 计算运动角度
    theta = math.acos(np.dot(ego_dxy, agent_dxy) / (np.linalg.norm(ego_dxy) * np.linalg.norm(agent_dxy) + 1e-8))

    # 计算系数
    v = 60 / 3.6  # 60 km/h，论文中没有说波速如何定义，此处用城市道路通常速度代替
    alpha_lon = max(0, (v + ego_v * math.cos(theta)) / (v - agent_v * math.cos(theta)))
    alpha_lat = math.exp(-beta * (math.sin(theta) ** 2))

    e = 0.5 * k * c * m * (ego_v - agent_v) ** 2 / distance
    e = alpha_lon * alpha_lat * e
    # return math.log(e + 1 + 1e-6)
    return e

def _fill_trainval_infos(nusc: NuScenes,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=6,
                         his_ts=2):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx

    risks = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        if not sample['scene_token'] in [*train_scenes, *val_scenes]:
            continue

        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        if sample['prev'] != '':
            sample_prev = nusc.get('sample', sample['prev'])
            sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
            pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
        else:
            pose_record_prev = None
        if sample['next'] != '':
            sample_next = nusc.get('sample', sample['next'])
            sd_rec_next = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
            pose_record_next = nusc.get('ego_pose', sd_rec_next['ego_pose_token'])
        else:
            pose_record_next = None

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        # 获取sample的can_bus信息
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        fut_valid_flag = True

        # 将sample深度拷贝为test_sample
        test_sample = copy.deepcopy(sample)
        # 找到6帧之后的数据，作为test_sample
        for i in range(fut_ts):
            if test_sample['next'] != '':
                test_sample = nusc.get('sample', test_sample['next'])
            else:
                fut_valid_flag = False

        # 记录信息
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'fut_valid_flag': fut_valid_flag,
            'map_location': map_location,
            # 'gt_descriptions': {}
        }

        # # 获取当前ego所在的道路类型
        # road_type, road_type_one_hot, road_type_all = get_road_type(nusc, map_location, pose_record)
        # info['gt_descriptions'].update({
        #     'road_type': road_type,
        #     'road_type_one_hot': road_type_one_hot.astype(np.float32),
        #     'road_type_all': road_type_all
        # })

        # 当 sample['next'] == '' 时，表示当前帧为最后一帧，需要将 frame_idx 重置为 0
        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        # 获取每帧数据的6个相机的数据
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat,
                e2g_t, e2g_r_mat, cam
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        # 获取当前数据的前几帧雷达数据（sweeps），并将这些数据存储在 sweeps 列表中
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(
                    nusc, sd_rec['prev'], l2e_t,
                    l2e_r_mat, e2g_t, e2g_r_mat, 'lidar'
                )
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # obtain annotation
        # 获取所有被标注物体的信息，包括物体属性和位姿信息
        # if not test:
        # 获取标注物信息
        annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

        # 获取物体的速度和有效性
        velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
        valid_flag = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 for anno in annotations], dtype=bool).reshape(-1)

        # 从boxes中获取定位、尺寸和姿态
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        # convert velo from global to lidar
        # 将物体的全局速度转化为雷达坐标系下的速度，个人理解为将ego设定为静止，计算其他物体的相对速度
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]
        
        # 获取物体的名称
        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in NuScenesDataset.NameMapping:
                # 规整名字，比如说，'human.pedestrian.adult' -> 'pedestrian'
                names[i] = NuScenesDataset.NameMapping[names[i]]
        names = np.array(names)

        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)  # [num_box, 7]
        assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'
        
        # get future coords for each box
        num_box = len(boxes)
        gt_fut_trajs = np.zeros((num_box, fut_ts, 2))  # [num_box, fut_ts, 2]
        gt_fut_yaw = np.zeros((num_box, fut_ts))  # [num_box, fut_ts]
        gt_fut_masks = np.zeros((num_box, fut_ts))  # [num_box, fut_ts]
        gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2)  # -(-rots - np.pi / 2 + np.pi / 2) = rots
        agent_lcf_feat = np.zeros((num_box, 9))  # [num_box, 9], (x, y, yaw, vx, vy, width, length, height, type)
        gt_fut_goal = np.zeros((num_box))
        # 遍历所有annotations
        for i, anno in enumerate(annotations):
            cur_box = boxes[i]
            cur_anno = anno
            agent_lcf_feat[i, 0:2] = cur_box.center[:2]	
            agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
            agent_lcf_feat[i, 3:5] = velocity[i]
            agent_lcf_feat[i, 5:8] = anno['size'] # width,length,height
            agent_lcf_feat[i, 8] = cat2idx[anno['category_name']] if anno['category_name'] in cat2idx.keys() else -1
            # 获取未来fut_ts帧的状态
            for j in range(fut_ts):
                if cur_anno['next'] != '':
                    anno_next = nusc.get('sample_annotation', cur_anno['next'])
                    box_next = Box(
                        anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation'])
                    )
                    # Move box to ego vehicle coord system.
                    box_next.translate(-np.array(pose_record['translation']))
                    box_next.rotate(Quaternion(pose_record['rotation']).inverse)
                    #  Move box to sensor coord system.
                    box_next.translate(-np.array(cs_record['translation']))
                    box_next.rotate(Quaternion(cs_record['rotation']).inverse)
                    gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]
                    gt_fut_masks[i, j] = 1
                    # add yaw diff，将四元素转换为欧拉角
                    _, _, box_yaw = quart_to_rpy([
                        cur_box.orientation.x, cur_box.orientation.y,
                        cur_box.orientation.z, cur_box.orientation.w
                    ])
                    _, _, box_yaw_next = quart_to_rpy([
                        box_next.orientation.x, box_next.orientation.y,
                        box_next.orientation.z, box_next.orientation.w
                    ])
                    gt_fut_yaw[i, j] = box_yaw_next - box_yaw
                    cur_anno = anno_next
                    cur_box = box_next
                else:
                    gt_fut_trajs[i, j:] = 0
                    break

            # get agent goal
            gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
            coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
            if coord_diff.max() < 1.0: # static
                gt_fut_goal[i] = 9
            else:
                box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

        # get ego history traj (offset format)
        # 获取车辆历史轨迹
        ego_his_trajs = np.zeros((his_ts+1, 3))
        ego_his_trajs_diff = np.zeros((his_ts+1, 3))
        sample_cur = sample
        for i in range(his_ts, -1, -1):
            if sample_cur is not None:
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_his_trajs[i] = pose_mat[:3, 3]
                has_prev = sample_cur['prev'] != ''
                has_next = sample_cur['next'] != ''
                if has_next:
                    sample_next = nusc.get('sample', sample_cur['next'])
                    pose_mat_next = get_global_sensor_pose(sample_next, nusc, inverse=False)
                    ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
                sample_cur = nusc.get('sample', sample_cur['prev']) if has_prev else None
            else:
                ego_his_trajs[i] = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
                ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]
        
        # global to ego at lcf
        ego_his_trajs = ego_his_trajs - np.array(pose_record['translation'])
        rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        # ego to lidar at lcf
        ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
        ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]

        # get ego futute traj (offset format)
        # 获取车辆未来轨迹
        ego_fut_trajs = np.zeros((fut_ts+1, 3))
        ego_fut_masks = np.zeros((fut_ts+1))
        sample_cur = sample
        for i in range(fut_ts+1):
            pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
            ego_fut_trajs[i] = pose_mat[:3, 3]
            ego_fut_masks[i] = 1
            if sample_cur['next'] == '':
                ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                break
            else:
                sample_cur = nusc.get('sample', sample_cur['next'])
        # global to ego at lcf
        ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
        rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        # ego to lidar at lcf
        ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

        # 至此，已经获取了ego和其他agent相对于ego的未来轨迹信息
        # xy：x代表了左右偏移量，y代表了向前偏移量
        # 接下来，需要根据这些信息生成比较合适的指令
        # 左转：x < -2，
        # 左偏转：-2 < x < -1
        # 直行：-1 < x < 1
        # 右偏转：1 < x < 2
        # 右转：2 < x

        # drive command according to final fut step offset from lcfw
        # 生成指令
        x_end = ego_fut_trajs[-1, 0]
        command = np.zeros(5)
        if x_end <= -2:
            command[0] = 1  # 左转
        elif -2 < x_end <= -1:
            command[1] = 1  # 左偏转
        elif -1 < x_end < 1:
            command[2] = 1  # 直行
        elif 1 <= x_end < 2:
            command[3] = 1  # 右偏转
        elif 2 <= x_end:
            command[4] = 1  # 右转
        else:
            raise ValueError('x_end out of range')

        # 获取当前道路交通情况
        # traffic_condition, traffic_condition_one_hot, traffic_condition_all = get_traffic_condition(agent_lcf_feat, names)

        # offset from lcf -> per-step offset
        ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

        ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
        ego_lcf_feat = np.zeros(9)
        # 根据odom推算自车速度及加速度
        _, _, ego_yaw = quart_to_rpy(pose_record['rotation'])
        ego_pos = np.array(pose_record['translation'])
        if pose_record_prev is not None:
            _, _, ego_yaw_prev = quart_to_rpy(pose_record_prev['rotation'])
            ego_pos_prev = np.array(pose_record_prev['translation'])
        if pose_record_next is not None:
            _, _, ego_yaw_next = quart_to_rpy(pose_record_next['rotation'])
            ego_pos_next = np.array(pose_record_next['translation'])
        assert (pose_record_prev is not None) or (pose_record_next is not None), 'prev token and next token all empty'
        if pose_record_prev is not None:
            ego_w = (ego_yaw - ego_yaw_prev) / 0.5
            ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
            ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
        else:
            ego_w = (ego_yaw_next - ego_yaw) / 0.5
            ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
            ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)

        ref_scene = nusc.get("scene", sample['scene_token'])
        try:
            pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
            steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_uts = [msg['utime'] for msg in steer_msgs]
            ref_utime = sample['timestamp']
            pose_index = locate_message(pose_uts, ref_utime)
            pose_data = pose_msgs[pose_index]
            steer_index = locate_message(steer_uts, ref_utime)
            steer_data = steer_msgs[steer_index]
            # initial speed
            v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
            # curvature (positive: turn left)
            steering = steer_data["value"]
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if map_location.startswith('singapore') else False
            if flip_flag:
                steering *= -1
            Kappa = 2 * steering / 2.588
        except:
            delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
            delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
            v0 = np.sqrt(delta_x**2 + delta_y**2)
            Kappa = 0

        ego_lcf_feat[:2] = np.array([ego_vx, ego_vy]) #can_bus[13:15]
        ego_lcf_feat[2:4] = can_bus[7:9]
        ego_lcf_feat[4] = ego_w #can_bus[12]
        ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
        ego_lcf_feat[7] = v0
        ego_lcf_feat[8] = Kappa

        # 根据轨迹计算风险值
        risk_values = np.zeros([num_box, fut_ts])
        # ego_fut_path = np.cumsum(ego_fut_trajs, axis=-2)
        # agent_fut_paths = np.cumsum(gt_fut_trajs, axis=-2) + agent_lcf_feat[..., None, :2]
        for i in range(num_box):
            for j in range(fut_ts):
                ego_dxy, agent_dxy = ego_fut_trajs[j, :2], gt_fut_trajs[i, j]
                agent_pose = agent_lcf_feat[i, None, :2]
                agent_name = names[i]
                distance = np.linalg.norm(np.cumsum(gt_fut_trajs[i, :(j+1)], axis=-2) + agent_pose - ego_dxy)
                risk_values[i, j] = compute_point_risk_value(ego_dxy, agent_dxy, agent_name, distance)
        risk_value_seq = np.sum(risk_values, axis=0)  # 每个时间步的总风险值
        risk_value = np.sum(risk_value_seq).item()  # 总风险值
        risk_value = math.log(risk_value + 1 + 1e-6)
        risks.append(risk_value)

        # # 可视化自车和他车轨迹
        # ego_fut_path = np.cumsum(ego_fut_trajs, axis=-2)
        # agent_fut_paths = np.cumsum(gt_fut_trajs, axis=-2) + agent_lcf_feat[..., None, :2]
        # fig = plt.figure()
        # plt.plot(ego_fut_path[:, 0], ego_fut_path[:, 1])
        # for i in range(fut_ts):
        #     plt.plot(ego_fut_path[i, 0], ego_fut_path[i, 1], 'ro', alpha=risk_value_seq[i]/(risk_value_seq.max() + 1e-8))
        #     plt.text(ego_fut_path[i, 0], ego_fut_path[i, 1], f'{risk_value_seq[i]:.2f}')
        # for i in range(num_box):
        #     plt.plot(agent_fut_paths[i, 0, 0], agent_fut_paths[i, 0, 1], 'k*')
        #     plt.plot(agent_fut_paths[i, :, 0], agent_fut_paths[i, :, 1], '--')
        # plt.axis('equal')
        # os.makedirs('image/risk_vis', exist_ok=True)
        # plt.savefig(f'image/risk_vis/{sample["token"]}.png')
        # plt.close(fig)

        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
        info['valid_flag'] = valid_flag
        info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
        info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
        info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
        info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
        info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
        info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
        info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
        info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
        info['gt_ego_fut_cmd'] = command.astype(np.float32)  # 指令
        info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)
        info['risk_value'] = np.array([risk_value], dtype=np.float32)
        # info['gt_descriptions'].update({
        #     'traffic_condition': traffic_condition,
        #     'traffic_condition_one_hot': traffic_condition_one_hot.astype(np.float32),
        #     'traffic_condition_all': traffic_condition_all
        # })

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    fig = plt.figure()
    plt.hist(risks, bins=100)
    plt.xlabel('Risk Value')
    plt.ylabel('Frequency')
    plt.title('Risk Distribution')
    os.makedirs('image/risk_vis', exist_ok=True)
    if test:
        plt.savefig(f'image/risk_vis/test_risks.png')
    else:
        plt.savefig(f'image/risk_vis/train_val_risks.png')
    plt.close(fig)
    print(f'Max risk value: {max(risks):.4f}, Min risk value: {min(risks):.4f}, Mean risk value: {np.mean(risks):.4f}, Median risk value: {np.median(risks):.4f}')

    return train_nusc_infos, val_nusc_infos

def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=False):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: False.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
