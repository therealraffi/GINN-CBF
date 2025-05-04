import einops
import torch
import numpy as np
import trimesh
import os
import random

from GINN.geometry.constraints import BoundingBox2DConstraint, CircleObstacle2D, CompositeInterface2D, Envelope2D, LineInterface2D, CompositeConstraint, SampleConstraint, SampleConstraintWithNormals, SampleEnvelope, SampleConstraintWithTargets, SampleConstraintinPairs, SampleConstraintsWithControls
from models.model_utils import tensor_product_xz
from models.point_wrapper import PointWrapper
from utils import get_is_out_mask
from visualization.utils_mesh import get_meshgrid_in_domain, get_mesh

from scipy.spatial import KDTree

mydir = "/scratch/rhm4nj/cral/ginn"

def calculate_bounds(points):
    x_min, x_max = torch.min(points[:, 0]), torch.max(points[:, 0])
    y_min, y_max = torch.min(points[:, 1]), torch.max(points[:, 1])
    z_min, z_max = torch.min(points[:, 2]), torch.max(points[:, 2])
    
    return torch.tensor([[x_min.item(), x_max.item()], [y_min.item(), y_max.item()], [z_min.item(), z_max.item()]]), (torch.tensor([x_min, y_min, z_min]), torch.tensor([x_max, y_max, z_max]))

class ProblemSampler():
    
    def __init__(self, config) -> None:
        self.config = config
        device = self.config['device']
        
        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []        
        self._obstacle_constraints = []
        
        if self.config['problem'] == 'simple_2d':
            self.config['bounds'] = torch.tensor([[-1, 1],[-0.5, 0.5]])  # [[x_min, x_max], [y_min, y_max]]
            envelope = Envelope2D(env_bbox=torch.tensor([[-.9, 0.9], [-0.4, 0.4]]), bounds=self.config['bounds'], device=device, sample_from=self.config['envelope_sample_from'])
            domain = BoundingBox2DConstraint(bbox=self.config['bounds'])
            
            l_target_normal = torch.tensor([-1.0, 0.0])
            r_target_normal = torch.tensor([1.0, 0.0])
            l_bc = LineInterface2D(start=torch.tensor([-.9, -.4]), end=torch.tensor([-.9, .4]), target_normal=l_target_normal)
            r_bc = LineInterface2D(start=torch.tensor([.9, -.4]), end=torch.tensor([.9, .4]), target_normal=r_target_normal)
            all_interfaces = CompositeInterface2D([l_bc, r_bc])
            
            circle_obstacle = CircleObstacle2D(center=torch.tensor([0.0, 0.0]), radius=torch.tensor(0.1))
            
            # sample once and keep the points; these are used for plotting
            self.constr_pts_dict = {
                'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope']).cpu().numpy().T,
                'interface': all_interfaces.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy().T,
                'obstacles': circle_obstacle.get_sampled_points(N=self.config['n_points_obstacles']).cpu().numpy().T,
                'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy().T,
            }
            
            # save the constraints
            self._envelope_constr = [envelope]
            self._interface_constraints = [l_bc, r_bc]
            self._obstacle_constraints = [circle_obstacle]
            self._domain = domain
            
        elif self.config['problem'] == 'pipes':
            self.config['bounds'] = torch.tensor([[-0.1, 1.6],[-0.1, 1.1]])  # [[x_min, x_max], [y_min, y_max]]
            # see paper page 15 - https://arxiv.org/pdf/2004.11797.pdf
            envelope = Envelope2D(env_bbox=torch.tensor([[0, 1.5],[0, 1]]), bounds=self.config['bounds'], device=device, sample_from=self.config['envelope_sample_from'])
            domain = BoundingBox2DConstraint(bbox=self.config['bounds'])
            
            l_target_normal = torch.tensor([-1.0, 0.0])
            r_target_normal = torch.tensor([1.0, 0.0])
            l_bc_1 = LineInterface2D(start=torch.tensor([0, 0.25 - 1/12]), end=torch.tensor([0, 0.25 + 1/12]), target_normal=l_target_normal)
            l_bc_2 = LineInterface2D(start=torch.tensor([0, 0.75 - 1/12]), end=torch.tensor([0, 0.75 + 1/12]), target_normal=l_target_normal)
            r_bc_1 = LineInterface2D(start=torch.tensor([1.5, 0.25 - 1/12]), end=torch.tensor([1.5, 0.25 + 1/12]), target_normal=r_target_normal)
            r_bc_2 = LineInterface2D(start=torch.tensor([1.5, 0.75 - 1/12]), end=torch.tensor([1.5, 0.75 + 1/12]), target_normal=r_target_normal)
            
            edge_in = 0.05
            upper_target_normal = torch.tensor([0.0, 1.0])
            lower_target_normal = torch.tensor([0.0, -1.0])
            l_bc_1_upper = LineInterface2D(start=torch.tensor([0, 0.25 + 1/12]), end=torch.tensor([edge_in, 0.25 + 1/12]), target_normal=upper_target_normal)
            l_bc_1_lower = LineInterface2D(start=torch.tensor([0, 0.25 - 1/12]), end=torch.tensor([edge_in, 0.25 - 1/12]), target_normal=lower_target_normal)
            l_bc_2_upper = LineInterface2D(start=torch.tensor([0, 0.75 + 1/12]), end=torch.tensor([edge_in, 0.75 + 1/12]), target_normal=upper_target_normal)
            l_bc_2_lower = LineInterface2D(start=torch.tensor([0, 0.75 - 1/12]), end=torch.tensor([edge_in, 0.75 - 1/12]), target_normal=lower_target_normal)
            
            r_bc_1_upper = LineInterface2D(start=torch.tensor([1.5, 0.25 + 1/12]), end=torch.tensor([1.5 - edge_in, 0.25 + 1/12]), target_normal=upper_target_normal)
            r_bc_1_lower = LineInterface2D(start=torch.tensor([1.5, 0.25 - 1/12]), end=torch.tensor([1.5 - edge_in, 0.25 - 1/12]), target_normal=lower_target_normal)
            r_bc_2_upper = LineInterface2D(start=torch.tensor([1.5, 0.75 + 1/12]), end=torch.tensor([1.5 - edge_in, 0.75 + 1/12]), target_normal=upper_target_normal)
            r_bc_2_lower = LineInterface2D(start=torch.tensor([1.5, 0.75 - 1/12]), end=torch.tensor([1.5 - edge_in, 0.75 - 1/12]), target_normal=lower_target_normal)
            
            all_interfaces = CompositeInterface2D([l_bc_1, l_bc_2, r_bc_1, r_bc_2,
                                                    l_bc_1_upper, l_bc_1_lower, l_bc_2_upper, l_bc_2_lower,
                                                    r_bc_1_upper, r_bc_1_lower, r_bc_2_upper, r_bc_2_lower,
                                                    ])
            
            # TODO: the obstacles are Decagons, not circles; probably not worth the effort though
            # the holes are described in the paper page 19, - https://arxiv.org/pdf/2004.11797.pdf
            circle_obstacle_1 = CircleObstacle2D(center=torch.tensor([0.5, 1.0/3]), radius=torch.tensor(0.05))
            circle_obstacle_2 = CircleObstacle2D(center=torch.tensor([0.5, 2.0/3]), radius=torch.tensor(0.05))
            circle_obstacle_3 = CircleObstacle2D(center=torch.tensor([1.0, 1.0/4]), radius=torch.tensor(0.05))
            circle_obstacle_4 = CircleObstacle2D(center=torch.tensor([1.0, 2.0/4]), radius=torch.tensor(0.05))
            circle_obstacle_5 = CircleObstacle2D(center=torch.tensor([1.0, 3.0/4]), radius=torch.tensor(0.05))
            all_obstacles = CompositeConstraint([circle_obstacle_1, circle_obstacle_2, 
                                                   circle_obstacle_3, circle_obstacle_4, circle_obstacle_5])
            
            # sample once and keep the points; these are used for plotting
            self.constr_pts_dict = {
                'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope']).cpu().numpy().T,
                'interface': all_interfaces.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy().T,
                'obstacles': all_obstacles.get_sampled_points(N=self.config['n_points_obstacles']).cpu().numpy().T,
                'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy().T,
            }
            
            # save the constraints
            self._envelope_constr = [envelope]
            self._interface_constraints = [l_bc_1, l_bc_2, r_bc_1, r_bc_2]
            self._obstacle_constraints = [all_obstacles]
            self._domain = domain
    
        elif self.config['problem'] == 'simjeb':
            # see paper page 5 - https://arxiv.org/pdf/2105.03534.pdf
            # measurements given in 100s of millimeters
            bounds = torch.from_numpy(np.load('GINN/simJEB/derived/bounds.npy')).to(device).float()
            
            # scale_factor and translation_vector
            scale_factor = np.load('GINN/simJEB/derived/scale_factor.npy')
            center_for_translation = np.load('GINN/simJEB/derived/center_for_translation.npy')
            
            # load meshes
            self.mesh_if = trimesh.load("GINN/simJEB/orig/interfaces.stl")
            self.mesh_env = trimesh.load("GINN/simJEB/orig/411_for_envelope.obj")
            
            # translate meshes
            self.mesh_if.apply_translation(-center_for_translation)
            self.mesh_env.apply_translation(-center_for_translation)
            
            # scale meshes
            self.mesh_if.apply_scale(1. / scale_factor)
            self.mesh_env.apply_scale(1. / scale_factor)
            
            # load points
            pts_far_outside_env = torch.from_numpy(np.load('GINN/simJEB/derived/pts_far_outside.npy')).to(device).float()
            pts_on_envelope = torch.from_numpy(np.load('GINN/simJEB/derived/pts_on_env.npy')).to(device).float()
            pts_inside_envelope = torch.from_numpy(np.load('GINN/simJEB/derived/pts_inside.npy')).to(device).float()
            pts_outside_envelope = torch.from_numpy(np.load('GINN/simJEB/derived/pts_outside.npy')).to(device).float()
            
            interface_pts = torch.from_numpy(np.load('GINN/simJEB/derived/interface_points.npy')).to(device).float()
            interface_normals = torch.from_numpy(np.load('GINN/simJEB/derived/interface_normals.npy')).to(device).float()
            pts_around_interface = torch.from_numpy(np.load('GINN/simJEB/derived/pts_around_interface_outside_env_10mm.npy')).to(device).float()
            # print(f'bounds: {bounds}')
            # print(f'pts_on_envelope: min x,y,z: {torch.min(pts_on_envelope, dim=0)[0]}, max x,y,z: {torch.max(pts_on_envelope, dim=0)[0]}')
            # print(f'pts_outside_envelope: min x,y,z: {torch.min(pts_outside_envelope, dim=0)[0]}, max x,y,z: {torch.max(pts_outside_envelope, dim=0)[0]}')
            # print(f'interface_pts: min x,y,z: {torch.min(interface_pts, dim=0)[0]}, max x,y,z: {torch.max(interface_pts, dim=0)[0]}')
            assert get_is_out_mask(pts_on_envelope, bounds).any() == False
            assert get_is_out_mask(interface_pts, bounds).any() == False
            
            self.config['bounds'] = bounds  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]        
            envelope = SampleEnvelope(pts_on_envelope=pts_on_envelope, pts_outside_envelope=pts_outside_envelope, sample_from=self.config['envelope_sample_from'])
            envelope_around_interface = SampleConstraint(sample_pts=pts_around_interface)
            pts_far_from_env_constraint = SampleConstraint(sample_pts=pts_far_outside_env)
            inside_envelope = SampleConstraint(sample_pts=pts_inside_envelope)
            domain = CompositeConstraint([inside_envelope])  ## TODO: test also with including outside envelope
            interface = SampleConstraintWithNormals(sample_pts=interface_pts, normals=interface_normals)

            self.constr_pts_dict = {
                # the envelope points are sampled uniformly from the 3 subsets
                'far_outside_envelope': pts_far_from_env_constraint.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
                'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
                'envelope_around_interface': envelope_around_interface.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
                # other constraints
                'interface': interface.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy(),
                'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy(),
            }

            self._envelope_constr = [envelope, envelope_around_interface, pts_far_from_env_constraint]
            self._interface_constraints = [interface]
            self._obstacle_constraints = None
            self._domain = domain

        elif self.config['problem'] == 'cube_hole':
            # see paper page 5 - https://arxiv.org/pdf/2105.03534.pdf
            # measurements given in 100s of millimeters
            mydir = config['dataset_dir']
            bounds = torch.from_numpy(np.load(f'{mydir}/bounds.npy')).to(device).float()
            
            # load points
            pts_far_outside_env = torch.from_numpy(np.load(f'{mydir}/pts_far_outside.npy')).to(device).float()
            pts_on_envelope = torch.from_numpy(np.load(f'{mydir}/pts_on_env.npy')).to(device).float()
            pts_inside_envelope = torch.from_numpy(np.load(f'{mydir}/pts_inside.npy')).to(device).float()
            pts_outside_envelope = torch.from_numpy(np.load(f'{mydir}/pts_outside.npy')).to(device).float()
            
            interface_pts = torch.from_numpy(np.load(f'{mydir}/interface_points.npy')).to(device).float()
            interface_normals = torch.from_numpy(np.load(f'{mydir}/interface_normals.npy')).to(device).float()
            pts_around_interface = torch.from_numpy(np.load(f'{mydir}/pts_around_interface_outside_env.npy')).to(device).float()

            pts_obstacle = torch.from_numpy(np.load(f'{mydir}/pts_obst.npy')).to(device).float()
            # print(f'bounds: {bounds}')
            # print(f'pts_on_envelope: min x,y,z: {torch.min(pts_on_envelope, dim=0)[0]}, max x,y,z: {torch.max(pts_on_envelope, dim=0)[0]}')
            # print(f'pts_outside_envelope: min x,y,z: {torch.min(pts_outside_envelope, dim=0)[0]}, max x,y,z: {torch.max(pts_outside_envelope, dim=0)[0]}')
            # print(f'interface_pts: min x,y,z: {torch.min(interface_pts, dim=0)[0]}, max x,y,z: {torch.max(interface_pts, dim=0)[0]}')
            assert get_is_out_mask(pts_on_envelope, bounds).any() == False
            assert get_is_out_mask(interface_pts, bounds).any() == False
            
            self.config['bounds'] = bounds  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]        
            envelope = SampleEnvelope(pts_on_envelope=pts_on_envelope, pts_outside_envelope=pts_outside_envelope, sample_from=self.config['envelope_sample_from'])
            envelope_around_interface = SampleConstraint(sample_pts=pts_around_interface)
            pts_far_from_env_constraint = SampleConstraint(sample_pts=pts_far_outside_env)
            inside_envelope = SampleConstraint(sample_pts=pts_inside_envelope)
            domain = CompositeConstraint([inside_envelope])  ## TODO: test also with including outside envelope
            interface = SampleConstraintWithNormals(sample_pts=interface_pts, normals=interface_normals)
            all_obstacles = SampleConstraint(sample_pts=pts_obstacle)

            self.constr_pts_dict = {
                # the envelope points are sampled uniformly from the 3 subsets
                'far_outside_envelope': pts_far_from_env_constraint.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
                'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
                'envelope_around_interface': envelope_around_interface.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
                # other constraints
                'interface': interface.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy(),
                'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy(),
                'obstacle': all_obstacles.get_sampled_points(N=self.config['n_points_obstacles']).cpu().numpy(),
            }
            
            self._envelope_constr = [envelope, pts_far_from_env_constraint, envelope_around_interface]
            self._interface_constraints = [interface]
            self._obstacle_constraints = [all_obstacles]
            self._domain = domain

        elif self.config['problem'] == 'grid_world':
            mydir = config['dataset_dir']

            # load points
            bounds_og = torch.from_numpy(np.load(os.path.join(mydir, "bounds.npy"))).to(device).float()
            scale_factor = torch.from_numpy(np.load(os.path.join(mydir, "scale_factor.npy"))).to(device).float()
            center_for_translation = torch.from_numpy(np.load(os.path.join(mydir, "center_for_translation.npy"))).to(device).float()

            pts_on_env = torch.from_numpy(np.load(os.path.join(mydir, "pts_on_env.npy"))).to(device).float()
            env_outside_pts = torch.from_numpy(np.load(os.path.join(mydir, "env_outside_pts.npy"))).to(device).float()
            domain_pts = torch.from_numpy(np.load(os.path.join(mydir, "pts_inside.npy"))).to(device).float()
            if_pts = torch.from_numpy(np.load(os.path.join(mydir, "interface_pts.npy"))).to(device).float()
            if_normals = torch.from_numpy(np.load(os.path.join(mydir, "interface_normals.npy"))).to(device).float()
            outer_pts = torch.from_numpy(np.load(os.path.join(mydir, "outside_points.npy"))).to(device).float()
            control_data = torch.from_numpy(np.load(os.path.join(mydir, "control_points.npy"))).to(device).float()
            control_points, controls = control_data[:, :3], control_data[:, 3:]

            outer_pts_dists = torch.from_numpy(np.load(os.path.join(mydir, "outside_points_dists.npy"))).to(device).float()
            pts_on_env_dists = torch.from_numpy(np.load(os.path.join(mydir, "pts_on_env_dists.npy"))).to(device).float()
            env_outside_pts_dists = torch.from_numpy(np.load(os.path.join(mydir, "env_outside_pts_dists.npy"))).to(device).float()

            control_data_interface = torch.from_numpy(np.load(os.path.join(mydir, "control_points_interface.npy"))).to(device).float()
            control_points_interface, controls_interface = control_data_interface[:, :3], control_data_interface[:, 3:]
            control_data_env = torch.from_numpy(np.load(os.path.join(mydir, "control_points_env.npy"))).to(device).float()
            control_points_env, controls_env = control_data_env[:, :3], control_data_env[:, 3:]
            control_data_on_env = torch.from_numpy(np.load(os.path.join(mydir, "control_points_on_env.npy"))).to(device).float()
            control_points_on_env, controls_on_env = control_data_on_env[:, :3], control_data_on_env[:, 3:]

            pts_on_env = (pts_on_env - center_for_translation) / scale_factor
            env_outside_pts = (env_outside_pts - center_for_translation) / scale_factor
            domain_pts = (domain_pts - center_for_translation) / scale_factor
            if_pts = (if_pts - center_for_translation) / scale_factor
            outer_pts = (outer_pts - center_for_translation) / scale_factor
            control_points = (control_points - center_for_translation) / scale_factor
            control_points_interface = (control_points_interface - center_for_translation) / scale_factor
            control_points_env = (control_points_env - center_for_translation) / scale_factor
            control_points_on_env = (control_points_on_env - center_for_translation) / scale_factor

            # outer_pts_dists = outer_pts_dists / scale_factor
            # pts_on_env_dists = pts_on_env_dists / scale_factor
            # env_outside_pts_dists = env_outside_pts_dists / scale_factor

            all_points = torch.vstack([domain_pts, pts_on_env, env_outside_pts, if_pts]) # dont include outer
            bounds, _ = calculate_bounds(all_points)
            bounds = bounds.to(device).float()

            # print("If pt", if_pts[0])
            # print("Outer pt", outer_pts[0])
            # print("Outer val", outer_pts_vals[0])

            # print("bounds_og", bounds_og, bounds_og.shape)
            # print("bounds", bounds, bounds.shape)

            # assert get_is_out_mask(env_outside_pts, bounds).any() == False
            assert get_is_out_mask(pts_on_env, bounds).any() == False
            assert get_is_out_mask(if_pts, bounds).any() == False

            self.config['bounds'] = bounds  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]        
            envelope = SampleEnvelope(pts_on_envelope=pts_on_env, pts_outside_envelope=env_outside_pts, sample_from=self.config['envelope_sample_from'])
            envelope_inner = SampleConstraint(sample_pts=pts_on_env)
            envelope_outside = SampleConstraint(sample_pts=env_outside_pts)
            outer = SampleConstraint(sample_pts=outer_pts)

            envelope_wtarget = SampleConstraintWithTargets(pts_on_env, pts_on_env_dists)
            envelope_outside_wtarget = SampleConstraintWithTargets(env_outside_pts, env_outside_pts_dists)
            outer_outside_wtarget = SampleConstraintWithTargets(outer_pts, outer_pts_dists)

            descent = SampleConstraintsWithControls(sample_pts=control_points, controls=controls)
            descent_env = SampleConstraintsWithControls(sample_pts=control_points_env, controls=controls_env)
            descent_on_env = SampleConstraintsWithControls(sample_pts=control_points_on_env, controls=controls_on_env)
            descent_interface = SampleConstraintsWithControls(sample_pts=control_points_interface, controls=controls_interface)

            domain = SampleConstraint(sample_pts=domain_pts)
            interface = SampleConstraintWithNormals(sample_pts=if_pts, normals=if_normals)

            self.constr_pts_dict = {
                'envelope': envelope.get_sampled_points(N=self.config['n_points_envelope']).cpu().numpy(),
                'far_outside_envelope': envelope_outside.get_sampled_points(N=self.config['n_points_envelope'] // 3).cpu().numpy(),
                'interface': interface.get_sampled_points(N=self.config['n_points_interfaces'])[0].cpu().numpy(),
                'domain': domain.get_sampled_points(N=self.config['n_points_domain']).cpu().numpy()
            }

            self._outer_constr = [envelope, envelope_outside, outer]
            self._outer_constr_counts = [.3, .3, .4]

            self._envelope_constr = [envelope_inner, envelope_outside]
            self._envelope_constr_wtargets = [envelope_wtarget, envelope_outside_wtarget]

            self._descent_constr = descent
            self._descent_env = descent_env
            self._descent_on_env = descent_on_env
            self._descent_interface = descent_interface

            self._interface_constraints = [interface]
            self._domain = domain
            self._obstacle_constraints = None

        else:
            raise NotImplementedError(f'Problem {self.config["problem"]} not implemented')
    
        self.bounds = config['bounds'].cpu()
        ## For plotting
        self.X0, self.X1, self.xs = get_meshgrid_in_domain(self.bounds)
        self.xs = torch.tensor(self.xs).float()
    
    def sample_from_envelope_inner(self, n=None):
        return self._envelope_constr[0].get_sampled_points(self.config['n_points_envelope'])

    def sample_from_envelope_outer(self, n=None):
        return self._envelope_constr[1].get_sampled_points(self.config['n_points_envelope'])

    def sample_from_envelope(self, n=None):
        pts_per_constraint = self.config['n_points_envelope'] // len(self._envelope_constr)
        return torch.cat([c.get_sampled_points(pts_per_constraint) for c in self._envelope_constr], dim=0)

    def sample_from_envelope_wtargets(self, n=None):
        pts_per_constraint = self.config['n_points_envelope'] // len(self._envelope_constr_wtargets)
        pts = []
        targets = []
        for c in self._envelope_constr_wtargets:
            p, t = c.get_sampled_points(pts_per_constraint)
            pts.append(p), targets.append(t)

        return torch.cat(pts, dim=0), torch.cat(targets, dim=0)

    def sample_from_outer(self): # tuple!
        return torch.cat([c.get_sampled_points(int(self.config['n_points_outer'] * self._outer_constr_counts[i])) for i, c in enumerate(self._outer_constr)], dim=0)

    # def sample_from_outer_only(self):
    #     return self._outer_only_constr.get_sampled_points(self.config['n_points_outer'])

    # def sample_for_descent(self): # tuple!
    #     def sample_from_face(global_min, global_max, fixed_axis, fixed_value):
    #         point = torch.empty(3)
    #         for i in range(3):
    #             if i == fixed_axis:
    #                 point[i] = fixed_value  # Fix the coordinate to the face value
    #             else:
    #                 point[i] = random.uniform(global_min[i], global_max[i])  # Random point in range
    #         return point

    #     def generate_surface_pairs(global_bounds, num_pairs):
    #         global_min = global_bounds[:, 0]
    #         global_max = global_bounds[:, 1]

    #         faces = [
    #             (0, global_min[0]),  # X-min
    #             (0, global_max[0]),  # X-max
    #             (1, global_min[1]),  # Y-min
    #             (1, global_max[1]),  # Y-max
    #             (2, global_min[2]),  # Z-min
    #             (2, global_max[2])   # Z-max
    #         ]

    #         def is_inside_bounds(point, bounds_list):
    #             for bounds in bounds_list:
    #                 min_vals, max_vals = bounds[:, 0], bounds[:, 1]
    #                 if torch.all((point >= min_vals) & (point <= max_vals)):  # Inside check
    #                     return True
    #             return False

    #         pairs = []
            
    #         while len(pairs) < num_pairs:
    #             face1, face2 = random.sample(faces, 2)
    #             point1 = sample_from_face(global_min, global_max, face1[0], face1[1])
    #             point2 = sample_from_face(global_min, global_max, face2[0], face2[1])
    #             pairs.append(torch.stack((point1, point2)))

    #         return torch.stack(pairs)
        
    #     pairs = generate_surface_pairs(self.config['bounds'], self.config['n_points_descent'])
    #     return pairs[0], pairs[1]

    def sample_for_descent(self): # tuple!
        return self._descent_constr.get_sampled_points(self.config['n_points_descent'])
    
    def sample_for_descent_interface(self): # tuple!
        return self._descent_interface.get_sampled_points(self.config['n_points_descent'])
    
    def sample_for_descent_env(self): # tuple!
        return self._descent_env.get_sampled_points(self.config['n_points_descent'])
    
    def sample_for_descent_on_env(self): # tuple!
        return self._descent_on_env.get_sampled_points(self.config['n_points_descent'])
    
    def sample_from_interface(self):
        pts_per_constraint = self.config['n_points_interfaces'] // len(self._interface_constraints)
        pts = []
        normals = []
        for c in self._interface_constraints:
            pts_i, normals_i = c.get_sampled_points(pts_per_constraint)
            pts.append(pts_i)
            normals.append(normals_i)
        return torch.cat(pts, dim=0), torch.cat(normals, dim=0)
    
    def sample_from_obstacles(self):
        pts_per_constraint = self.config['n_points_obstacles'] // len(self._obstacle_constraints)
        return torch.vstack([c.get_sampled_points(pts_per_constraint) for c in self._obstacle_constraints])
    
    def sample_from_domain(self):
        return self._domain.get_sampled_points(self.config['n_points_domain'])
    
    def recalc_output(self, f, params, z_latents):
        """Compute the function on the grid.
        epoch: will be used to identify figures for wandb or saving
        :param z_latents: 
        """
        with torch.no_grad():
            if self.config['nx']==2:
                y = f(params, *tensor_product_xz(self.xs, z_latents)).detach().cpu().numpy()
                Y = einops.rearrange(y, '(bz h w) 1 -> bz h w', bz=len(z_latents), h=self.X0.shape[0])
                return y, Y
            elif self.config['nx']==3:
                meshes = []
                for z_ in z_latents: ## do marching cubes for every z
                    
                    def f_fixed_z(x):
                        """A wrapper for calling the model with a single fixed latent code"""
                        return f(params, *tensor_product_xz(x, z_.unsqueeze(0))).squeeze(0)
                    
                    verts_, faces_ = get_mesh(f_fixed_z,
                                                N=self.config["mc_resolution"],
                                                device=z_latents.device,
                                                bbox_min=self.config["bounds"][:,0],
                                                bbox_max=self.config["bounds"][:,1],
                                                chunks=1,
                                                return_normals=0)
                    # print(f"Found a mesh with {len(verts_)} vertices and {len(faces_)} faces")
                    meshes.append((verts_, faces_))
                return meshes
        
    def is_inside_envelope(self, p_np: PointWrapper):
        """Remove points that are outside the envelope"""
        valid_problems = ['simjeb', 'cube_hole', 'pt_cld', 'grid_world']
        if not self.config['problem'] in valid_problems:
            raise NotImplementedError('This function is only implemented for the simjeb problem')

        if self.config['problem'] == 'simjeb':
            is_inside_mask = self.mesh_env.contains(p_np.data)
            return is_inside_mask
        else:
            ptcld = np.vstack((self.constr_pts_dict['domain'], self.constr_pts_dict['envelope']))

            # np.save("ptcld.npy", ptcld)
            # np.save("inp_ptcld.npy", p_np.data)

            tolerance = 0.1  # Adjust as needed
            kdtree = KDTree(ptcld)
            distances, _ = kdtree.query(p_np.data, distance_upper_bound=tolerance)
            inside_points_mask = distances < tolerance
            inside_points = torch.tensor(inside_points_mask, dtype=torch.bool).cpu().numpy()
            return inside_points