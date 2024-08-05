import numpy as np
#import carla
from gym import spaces
import cv2 as cv
from collections import deque
from pathlib import Path
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2,Point2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from gym_navsim.core.obs_manager.obs_manager import ObsManagerBase
from navsim.common.enums import BoundingBoxIndex
from nuplan.common.actor_state.car_footprint import CarFootprint
#from carla_gym.utils.traffic_light import TrafficLightHandler
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from typing import Any, Dict, List
from shapely import affinity
from shapely.geometry import Polygon, LineString
from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)

class ObsManager(ObsManagerBase):
    def __init__(self, obs_configs):
        self._width = int(obs_configs['width_in_pixels'])
        self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']
        self._pixels_per_meter = obs_configs['pixels_per_meter']
        self._history_idx = obs_configs['history_idx']
        self._scale_bbox = obs_configs.get('scale_bbox', True)
        self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)

        self._history_queue = deque(maxlen=20)

        self._image_channels = 3
        self._masks_channels = 3 + 3*len(self._history_idx)
        self._parent_actor = None
        self._world = None

        #self._map_dir = Path(__file__).resolve().parent / 'maps'
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {'rendered': spaces.Box(
                low=0, high=255, shape=(self._width, self._width, self._image_channels),
                dtype=np.uint8),
             'masks': spaces.Box(
                low=0, high=255, shape=(self._masks_channels, self._width, self._width),
                dtype=np.uint8)})

    def attach_ego_vehicle(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        """
    @staticmethod
    def _get_stops(criteria_stop):
        stop_sign = criteria_stop._target_stop_sign
        stops = []
        if (stop_sign is not None) and (not criteria_stop._stop_completed):
            bb_loc = carla.Location(stop_sign.trigger_volume.location)
            bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            trans = stop_sign.get_transform()
            stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
        return stops
    """
    def get_observation(self,time):
        # Şu an zero zero
        ev_loc = Point2D(32,32)
        ev_rot = 0
        self.scene = self.ego_vehicle.scene
        start_idx = self.scene.scene_metadata.num_history_frames + time - 1
        human_trajectory_arr = self.ego_vehicle.human_trajectory
        trajectory_arr = self.ego_vehicle.trajectory
        for history_idx in self._history_idx:
            past_idx = start_idx + history_idx
            frame = self.scene.frames[past_idx]
            # Traffic lights 
            tl_green = []
            tl_red = []
            traffic_lights = frame.traffic_lights
            origin = StateSE2(*frame.ego_status.ego_pose)
            """
            initial_ego_state = self.ego_vehicle.metric_cache.ego_state
            if self.ego_vehicle.trajectory is None:
                origin = StateSE2(*frame.ego_status.ego_pose)
            else:   
                # Hardcodelu şimdilik
                pred_sampling = TrajectorySampling(num_poses=len(trajectory_arr[3:]),interval_length=0.5)
                pred_traj = Trajectory(poses=trajectory_arr[3:],trajectory_sampling=pred_sampling)
                pred_trajectory = transform_trajectory(pred_traj,initial_ego_state)
                pred_states = get_trajectory_as_array(pred_trajectory, pred_sampling, initial_ego_state.time_point)
                #print("Trajectory :",trajectory_arr)
                origin = StateSE2(*pred_states[time][:3])
            """
            map_object_dict = self.scene.map_api.get_proximal_map_objects(
                point=origin.point,
                radius=max((64,64)),
                layers=[SemanticMapLayer.LANE_CONNECTOR],
            )
            # No traffic lights for now
            for map_object in map_object_dict[SemanticMapLayer.LANE_CONNECTOR]:
                for idx,state in traffic_lights:
                    # Nuplan warning burada büyük ihtimal
                    if idx == int(map_object.id):
                        polygon: Polygon = self._geometry_local_coords(map_object.polygon, origin)
                        # Burada da değiştirdim
                        poligon_coords = np.array([[x,y] for x,y in polygon.exterior.coords])
                        for i in range(len(polygon.exterior.coords)):
                            angle_diff = trajectory_arr[start_idx][2] - human_trajectory_arr[start_idx][2]
                            according_to_human = self.rotate(trajectory_arr[start_idx,:2] - human_trajectory_arr[start_idx,:2],-human_trajectory_arr[start_idx,2])
                            poligon_coords[i] -= according_to_human
                            poligon_coords[i] = self.rotate(poligon_coords[i],-angle_diff)
                        polygon = Polygon(poligon_coords)
                        """
                        x,y = linestring.xy
                        linestring_coords = np.array([x,y]).T
                        for i in range(len(linestring_coords)):
                            angle_diff = trajectory_arr[start_idx][2] - human_trajectory_arr[past_idx][2]
                            according_to_human = self.rotate(trajectory_arr[start_idx,:2] - human_trajectory_arr[past_idx,:2],-human_trajectory_arr[past_idx,2])
                            linestring_coords[i] -= according_to_human
                            linestring_coords[i] = self.rotate(linestring_coords[i][:2],-angle_diff)
                            points = [Point2D(x,y) for x,y in linestring_coords]
                            #linestring = LineString(points[:10])
                            linestring = LineString(points)
                            if state:
                                tl_green.append(linestring)
                            else:
                                tl_red.append(linestring)
                        """
                        if state:
                            tl_green.append(polygon)
                        else:
                            tl_red.append(polygon)
            # Vehicle and walker bounding boxes
            vehicle_bbox_list = []
            walker_bbox_list = []
            for i in range(len(frame.annotations.boxes)):
                box = frame.annotations.boxes[i]
                x, y, heading = (
                    box[BoundingBoxIndex.X],
                    box[BoundingBoxIndex.Y],
                    box[BoundingBoxIndex.HEADING],
                )
                angle_diff = trajectory_arr[start_idx][2] - human_trajectory_arr[past_idx][2]
                agent_according_to_human = self.rotate(trajectory_arr[start_idx,:2] - human_trajectory_arr[past_idx,:2],-human_trajectory_arr[past_idx,2])
                xy = np.array([x,y])
                xy -= agent_according_to_human
                xy = self.rotate(xy,-angle_diff)
                heading -= angle_diff
                box_length, box_width, box_height = box[3], box[4], box[5]
                agent_box = OrientedBox(StateSE2(xy[0], xy[1], heading), box_length, box_width, box_height)
                if frame.annotations.names[i] == "vehicle":
                    vehicle_bbox_list.append(agent_box)
                elif frame.annotations.names[i] == "pedestrian":
                    walker_bbox_list.append(agent_box)
            """
            tl_green = TrafficLightHandler.get_stopline_vtx(ev_loc, 0)
            tl_yellow = TrafficLightHandler.get_stopline_vtx(ev_loc, 1)
            tl_red = TrafficLightHandler.get_stopline_vtx(ev_loc, 2)
            stops = self._get_stops(self._parent_actor.criteria_stop)
            """
            self._history_queue.append((vehicle_bbox_list, walker_bbox_list, tl_green, [], tl_red, []))
        #self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        # objects with history
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
            = self._get_history_masks(M_warp)
            # layers for plotting complete layers
        polygon_layers: List[SemanticMapLayer] = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.CROSSWALK,
        ]

        # layers for plotting complete layers
        polyline_layers: List[SemanticMapLayer] = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
        ]
        frame = self.scene.frames[start_idx]
        initial_ego_state = self.ego_vehicle.metric_cache.ego_state
        # Hardcodelu şimdilik
        #print("Trajectory :",trajectory_arr)
        # Hatalı ?
        #origin = StateSE2(*(trajectory_arr[start_idx][:3] + np.array([*initial_ego_state.center])))
        #print("Origin",pred_states[time][:3],asil_traj[time])
        origin = StateSE2(*frame.ego_status.ego_pose)
        map_object_dict = self.scene.map_api.get_proximal_map_objects(
            point=origin.point,
            radius=max((64,64)),
            layers=list(set(polygon_layers + polyline_layers)),
        )
        
        road_layers : List[SemanticMapLayer] = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.INTERSECTION,
        ]
        road_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for layer in road_layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, origin)
                # Burada da değiştirdim
                poligon_coords = np.array([[x,y] for x,y in polygon.exterior.coords])
                for i in range(len(polygon.exterior.coords)):
                    xy = np.array([x,y])
                    angle_diff = trajectory_arr[start_idx][2] - human_trajectory_arr[start_idx][2]
                    according_to_human = self.rotate(trajectory_arr[start_idx,:2] - human_trajectory_arr[start_idx,:2],-human_trajectory_arr[start_idx,2])
                    poligon_coords[i] -= according_to_human
                    poligon_coords[i] = self.rotate(poligon_coords[i],-angle_diff)
                polygon = Polygon(poligon_coords)
                self.add_polygon_to_image(road_mask, polygon,M_warp)
        road_mask = road_mask.astype(bool)

        lane_mask_all = np.zeros([self._width, self._width], dtype=np.uint8)
        for layer in polyline_layers:
            for map_object in map_object_dict[layer]:
                    linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, origin)
                    # Burada değiştirdim
                    x,y = linestring.xy
                    linestring_coords = np.array([x,y]).T
                    for i in range(len(linestring_coords)):
                        angle_diff = trajectory_arr[start_idx][2] - human_trajectory_arr[start_idx][2]
                        according_to_human = self.rotate(trajectory_arr[start_idx,:2] - human_trajectory_arr[start_idx,:2],-human_trajectory_arr[start_idx,2])
                        linestring_coords[i] -= according_to_human
                        linestring_coords[i] = self.rotate(linestring_coords[i][:2],-angle_diff)
                        points = [Point2D(x,y) for x,y in linestring_coords]
                        linestring = LineString(points)
                    self.add_linestring_to_image(lane_mask_all, linestring,M_warp,thickness=1)
        lane_mask_all = lane_mask_all.astype(bool)
        
        rotated_pdm_points = self.convert_relative_trajectories(self.ego_vehicle.route[:,:2],start_idx,trajectory_arr,human_trajectory_arr,3)
        pdm_points = [Point2D(x,y) for x,y in rotated_pdm_points]
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(x)[::-1]] for x in pdm_points])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        route_mask = route_mask.astype(bool)
        """
        # road_mask, lane_mask
        lane_mask_all = cv.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_broken = cv.warpAffine(self._lane_marking_white_broken, M_warp,
                                         (self._width, self._width)).astype(np.bool)

        # route_mask
        route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
        route_in_pixel = np.array([[self._world_to_pixel(wp.transform.location)]
                                   for wp, _ in self._parent_actor.route_plan[0:80]])
        route_warped = cv.transform(route_in_pixel, M_warp)
        cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
        route_mask = route_mask.astype(np.bool)
        """
        # ev_mask
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(0, 0, 0),
            vehicle_parameters=get_pacifica_parameters(),
        )
        ev_mask = self._get_mask_from_actor_list([car_footprint.oriented_box], M_warp)
        ev_mask_col = self._get_mask_from_actor_list([car_footprint.oriented_box], M_warp,scale=self._scale_mask_col)
        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[route_mask] = COLOR_ALUMINIUM_3
        image[lane_mask_all] = COLOR_MAGENTA
        # Debug için pdm points 
        pdm_points = np.array(self.ego_vehicle.route) # Copying it to not make changes in the original
        for i in range(len(pdm_points)):
            #angle_diff = trajectory_arr[start_idx][2] - human_trajectory_arr[3][2]
            #according_to_human = self.rotate(trajectory_arr[start_idx,:2] - human_trajectory_arr[3,:2],-human_trajectory_arr[3,2])
            #pdm_points[i][:2] -= according_to_human
            pdm_points[i][:2] -= trajectory_arr[start_idx,:2]
            pdm_points[i][:2] = self.rotate(pdm_points[i][:2],-trajectory_arr[start_idx,2])
            #pdm_points[i][:2] = self.rotate(pdm_points[i][:2],-angle_diff)
        pdm_points = [Point2D(x,y) for x,y in pdm_points[:,:2]]
        pdm_in_pixel = np.array([[self._world_to_pixel(x)[::-1]] for x in pdm_points])
        pdm_warped = cv.transform(pdm_in_pixel, M_warp)
        for point in pdm_warped:
            cv.circle(image, tuple(map(int,point[0])), 3, (0,255,0), -1)
        """
        image[lane_mask_broken] = COLOR_MAGENTA_2
        """
        h_len = len(self._history_idx)
        for i, mask in enumerate(stop_masks):
            image[mask] = tint(COLOR_YELLOW_2, (h_len - i)*0.2)
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len - i)*0.2)
        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len - i)*0.2)
        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len - i)*0.2)
        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len - i)*0.2)
        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len - i)*0.2)
        image[ev_mask] = COLOR_WHITE
 
        for mask in (*stop_masks, *tl_green_masks, *tl_yellow_masks, *tl_red_masks, *vehicle_masks, *walker_masks,ev_mask):
            mask = mask.astype(np.uint8)
        # image[obstacle_mask] = COLOR_BLUEC
        
        # masks
        c_road = road_mask.astype(np.uint8) * 255
        c_route = route_mask.astype(np.uint8) * 255
        c_lane = lane_mask_all.astype(np.uint8) * 255
        """
        c_lane[lane_mask_broken] = 120
        """
        # masks with history
        c_tl_history = []
        for i in range(len(self._history_idx)):
            c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
            c_tl[tl_green_masks[i]] = 80
            c_tl[tl_yellow_masks[i]] = 170
            c_tl[tl_red_masks[i]] = 255
            c_tl[stop_masks[i]] = 255
            c_tl_history.append(c_tl)
        c_vehicle_history = [m*255 for m in vehicle_masks]
        c_walker_history = [m*255 for m in walker_masks]
        masks = np.stack(( c_road,c_route,c_lane,*c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
        masks = np.transpose(masks, [2, 0, 1])
        for i,mask in enumerate(masks):
            mask = mask.astype(np.uint8)
        cv.putText(image, f"Time: {time}", (40, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        obs_dict = {'rendered': image, 'masks': masks}
        #self._parent_actor.collision_px = np.any(ev_mask_col & walker_masks[-1])
        self.ego_vehicle.collision_px = np.any(ev_mask & walker_masks[-1])
        self.ego_vehicle.collision_px = np.any(ev_mask & vehicle_masks[-1])

        return obs_dict

    def _get_history_masks(self, M_warp):
        qsize = len(self._history_queue)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
        for idx in self._history_idx:
            idx += len(self._history_queue) - 1
            vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]
            vehicle_masks.append(self._get_mask_from_actor_list(vehicles, M_warp))
            walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp))
            tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, M_warp))
            tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, M_warp))
            tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, M_warp))
            stop_masks.append(self._get_mask_from_actor_list(stops, M_warp))
        return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks

    def _get_mask_from_stopline_vtx(self, stopline_vtx, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for linestring in stopline_vtx:
                self.add_polygon_to_image(mask, linestring,M_warp)
                #self.add_linestring_to_image(mask, linestring,M_warp,thickness=1)
                """
                xs,ys = linestring.xy
                points = [Point2D(x,y) for x,y in zip(xs,ys)]
                corners_in_pixel = np.array([[self._world_to_pixel(corner)[::-1]] for corner in points])
                corners_warped = cv.transform(corners_in_pixel.round(), M_warp)
                # Adding a huge circle
                cv.circle(mask, tuple(map(int, corners_warped[0, 0])), 8, 1, -1)
                """
            
        return mask.astype(bool)

    def _get_mask_from_actor_list(self, actor_list, M_warp,scale=1):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for bounding_box in actor_list:
            corners = bounding_box.all_corners()
            corners_in_pixel = np.array([[self._world_to_pixel(corner)[::-1] * scale] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)
            hull = cv.convexHull(np.round(corners_warped).astype(np.int32))
            if hull.shape[1] != 1:
                hull = hull.reshape(-1, 1, 2)
            cv.fillConvexPoly(mask, hull, 1)
        return mask.astype(bool)

    def _get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = 0 # ev_rot kullanmıyorum şu anık # np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

        bottom_left = [32,-32]#ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
        top_left = [32,32]#ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
        top_right = [-32,32]#ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec
        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width-1],
                            [0, 0],
                            [self._width-1, 0]], dtype=np.float32)
        return cv.getAffineTransform(src_pts, dst_pts)

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * location.x #(location.x - self._world_offset[0])
        y = self._pixels_per_meter * location.y #(location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return self._pixels_per_meter * width

    def clean(self):
        self._history_queue.clear()

    def _geometry_local_coords(self,geometry: Any, origin: StateSE2) -> Any:
        """ Helper for transforming shapely geometry in coord-frame """
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y
        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
        return rotated_geometry
                
    def add_polygon_to_image(self, image: np.ndarray, polygon: Polygon,M_warp) -> None:
        """
        Adds shapely polygon to an OpenCV image by setting the required pixels to one.
        :param image: OpenCV image as a numpy array
        :param polygon: shapely Polygon 
        :return: image with polygon pixels set to one
        """
        
        def _add_element_helper(element: Polygon):

            """ Helper to add single polygon to image """
            exterior_points = [Point2D(x,y) for x,y in element.exterior.coords]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)[::-1]] for corner in exterior_points])
            corners_warped = cv.transform(corners_in_pixel, M_warp)
            hull = cv.convexHull(np.round(corners_warped).astype(np.int32))
            if hull.shape[1] != 1:
                hull = hull.reshape(-1, 1, 2)
            cv.fillConvexPoly(image, hull, 1)
            for interior in element.interiors:
                interior_points= [Point2D(x,y) for x,y in interior.coords]
                corners_in_pixel = np.array([[self._world_to_pixel(corner)[::-1]] for corner in interior_points])
                corners_warped = cv.transform(corners_in_pixel, M_warp)
                hull = cv.convexHull(np.round(corners_warped).astype(np.int32))
                if hull.shape[1] != 1:
                    hull = hull.reshape(-1, 1, 2)
                cv.fillConvexPoly(image, hull, 1)
            

        if isinstance(polygon, Polygon):
            _add_element_helper(polygon)
        else:
            # NOTE: in rare cases, a map polygon has several sub-polygons.
            for element in polygon:
                _add_element_helper(element)
    def add_linestring_to_image(self,image: np.ndarray, linestring: LineString,M_warp,thickness=1) -> None:
        xs,ys = linestring.xy
        points = [Point2D(x,y) for x,y in zip(xs,ys)]
        corners_in_pixel = np.array([[self._world_to_pixel(corner)[::-1]] for corner in points])
        corners_warped = cv.transform(corners_in_pixel.round(), M_warp)
        for i in range(len(corners_warped)-1):
            pt1 = tuple(map(int, corners_warped[i, 0]))
            pt2 = tuple(map(int, corners_warped[i+1, 0]))
            cv.line(image, pt1, pt2, color=1, thickness=thickness)
    def rotate(self, points, angle):
        """Rotate points by a given angle."""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return points @ rotation_matrix.T
    def convert_relative_trajectories(self,points,start_idx,want_arr,real_arr,past_idx=None):
        if past_idx is None:
            past_idx = start_idx
        res = []
        angle_diff = want_arr[start_idx][2] - real_arr[past_idx][2]
        agent_according_to_human = self.rotate(want_arr[start_idx,:2] - real_arr[past_idx,:2],-real_arr[past_idx,2])
        for x,y in points:
            xy = np.array([x,y])
            xy -= agent_according_to_human
            xy = self.rotate(xy,-angle_diff)
            res.append(xy)
        return np.array(res)

        
