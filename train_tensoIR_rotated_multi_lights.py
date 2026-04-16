import os
import sys

import torch
from tqdm.auto import tqdm
from opt import config_parser
import datetime

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from renderer import * 
from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from utils.s3im_loss import S3IM
from dataLoader import dataset_dict



args = config_parser()
print(args)


# ---------------------------------------------------------------------------
# Curriculum scheduling helpers
# ---------------------------------------------------------------------------

def smoothstep(iteration, start, end):
    """Smooth 0→1 ramp between [start, end]. Returns float in [0, 1]."""
    if iteration <= start:
        return 0.0
    if iteration >= end:
        return 1.0
    t = (iteration - start) / (end - start)
    return t * t * (3.0 - 2.0 * t)


def sigmoid_ramp(iteration, ramp_start, ramp_end):
    """Sigmoid-shaped 0→1 ramp centred at midpoint of [ramp_start, ramp_end]."""
    if iteration <= ramp_start:
        return 0.0
    if iteration >= ramp_end:
        return 1.0
    mid = (ramp_start + ramp_end) / 2.0
    scale = (ramp_end - ramp_start) / 8.0  # ±4σ spans the interval
    return float(torch.sigmoid(torch.tensor((iteration - mid) / scale)).item())

# Setup multi-device training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()

device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu")
print(f'Running with {num_gpus} GPU(s)...')

renderer = Renderer_TensoIR_train



class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]



@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensoIR = eval(args.model_name)(**kwargs)
    tensoIR.load(ckpt)

    alpha, _ = tensoIR.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensoIR.aabb.cpu(), level=0.005)



def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, args.hdrdir, light_rotation=args.light_rotation, split='test', downsample=args.downsample_train, is_stack=False,
                           sub=args.test_number)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensoIR = eval(args.model_name)(**kwargs)
    tensoIR.load(ckpt)



    logfolder = f'{args.basedir}/test_{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'

    # if args.render_train:
    #     os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    #     train_dataset = dataset(args.datadir, args.hdrdir, split='train', downsample=args.downsample_train,
    #                             is_stack=False)
    #     evaluation_all(train_dataset, tensoIR, args, renderer,visibility_net, f'{logfolder}/imgs_test_all/',
    #                    N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device, test_all=True)

    if args.render_test:
        PSNRs_test, PSNRs_rgb_brdf_test, MAE_test,\
            PSNR_albedo_single, PSNR_albedo_three = evaluation_iter_TensoIR(
                                                            test_dataset, 
                                                            tensoIR, 
                                                            args, 
                                                            renderer, 
                                                            f'{logfolder}/imgs_test_all/', 
                                                            N_samples=-1,
                                                            white_bg=white_bg, 
                                                            ndc_ray=ndc_ray, 
                                                            device=device,
                                                            test_all=True
                                                            )
        print(f'PSNRs_test: {PSNRs_test}')
        print(f'PSNRs_rgb_brdf_test: {PSNRs_rgb_brdf_test}')
        print(f'MAE_test: {MAE_test}')
        print(f'PSNR_albedo_single: {PSNR_albedo_single}')
        print(f'PSNR_albedo_three: {PSNR_albedo_three}')
    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset, tensoIR, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
    #                     N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
                            args.datadir, 
                            args.hdrdir, 
                            split='train', 
                            downsample=args.downsample_train, 
                            light_name=args.light_name,
                            light_rotation=args.light_rotation
                            )
    test_dataset = dataset(
                            args.datadir, 
                            args.hdrdir, 
                            split='test', 
                            downsample=args.downsample_test, 
                            light_name=args.light_name,
                            light_rotation=args.light_rotation
                            )
    # if is_distributed:
    #     train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
    #                                                         rank=dist.get_rank())
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, sampler=train_sampler,
    #                                                    num_workers=16, drop_last=True, pin_memory=True)
    # else:
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, num_workers=16,
    #                                                    drop_last=True, shuffle=True)


    print(f'Finish reading dataset')

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/checkpoints', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # copy the config file into the log folder
    os.system(f'cp {args.config} {logfolder}')

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)  # number of voxels in each direction
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensoIR = eval(args.model_name)(**kwargs)
        tensoIR.load(ckpt)
    else:
        tensoIR = eval(args.model_name)(aabb, 
                                        reso_cur, 
                                        device,
                                        density_n_comp=n_lamb_sigma, 
                                        appearance_n_comp=n_lamb_sh,
                                        app_dim=args.data_dim_color, 
                                        near_far=near_far,
                                        shadingMode=args.shadingMode, 
                                        alphaMask_thres=args.alpha_mask_thre,
                                        density_shift=args.density_shift, 
                                        distance_scale=args.distance_scale,
                                        pos_pe=args.pos_pe, 
                                        view_pe=args.view_pe, 
                                        fea_pe=args.fea_pe,
                                        featureC=args.featureC, 
                                        step_ratio=args.step_ratio,
                                        fea2denseAct=args.fea2denseAct,
                                        normals_kind = args.normals_kind,
                                        light_rotation=args.light_rotation,
                                        light_kind=args.light_kind,
                                        dataset=train_dataset,
                                        numLgtSGs = args.numLgtSGs,
                                        )



    # --- Stage 2: Freeze Geometry and Environmental Lighting Field ---
    # Skipped when unified_curriculum=True — all parameters stay trainable end-to-end.
    if args.stage == 2 and not args.unified_curriculum:
        print("--- STAGE 2: Freezing Geometry and Environmental Lighting Field ---")
        # Freeze the spatial tensor grids (Density & Appearance)
        for param in tensoIR.density_line.parameters():
            param.requires_grad = False
        for param in tensoIR.density_plane.parameters():
            param.requires_grad = False
        for param in tensoIR.app_line.parameters():
            param.requires_grad = False
        for param in tensoIR.app_plane.parameters():
            param.requires_grad = False

        # Freeze the basis matrix and light embedding
        for param in tensoIR.basis_mat.parameters():
            param.requires_grad = False
        for param in tensoIR.light_line.parameters():
            param.requires_grad = False

        # Freeze the environment lighting parameters
        if tensoIR.light_kind == 'sg':
            tensoIR.lgtSGs.requires_grad = False
        elif tensoIR.light_kind == 'pixel':
            tensoIR._light_rgbs.requires_grad = False

        # Freeze the Stage 1 Color MLP
        if isinstance(tensoIR.renderModule, torch.nn.Module):
            for param in tensoIR.renderModule.parameters():
                param.requires_grad = False

        # Ensure BRDF MLP IS trainable
        if isinstance(tensoIR.renderModule_brdf, torch.nn.Module):
            for param in tensoIR.renderModule_brdf.parameters():
                param.requires_grad = True

        # Ensure Normal MLP IS trainable (if it exists)
        if hasattr(tensoIR, 'renderModule_normal') and isinstance(tensoIR.renderModule_normal, torch.nn.Module):
            for param in tensoIR.renderModule_normal.parameters():
                param.requires_grad = True

        # Print summary of trainable vs frozen parameters
        total_params = sum(p.numel() for p in tensoIR.parameters())
        trainable_params = sum(p.numel() for p in tensoIR.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Frozen params: {frozen_params:,}")

    grad_vars = tensoIR.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / (args.lr_decay_iters))
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / (args.lr_decay_iters))

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)


    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(
        torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs_test, PSNRs_rgb_brdf_test = [0], [0]
    PSNRs_rgb, PSNRs_rgb_brdf = [], []

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    # S3IM loss
    s3im_loss_fn = None
    if args.s3im_weight > 0:
        s3im_loss_fn = S3IM(
            kernel_size=args.s3im_kernel_size,
            stride=args.s3im_stride,
            repeat_time=args.s3im_repeat_time,
            patch_height=args.s3im_patch_height,
            patch_width=args.s3im_patch_width,
        ).to(device)
        print(f"S3IM loss enabled with weight={args.s3im_weight}, "
              f"kernel_size={args.s3im_kernel_size}, stride={args.s3im_stride}, "
              f"repeat_time={args.s3im_repeat_time}, "
              f"patch={args.s3im_patch_height}x{args.s3im_patch_width}")


    all_rays, all_rgbs, all_masks, all_light_idx = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_masks, train_dataset.all_light_idx
    # Filter rays outside the bbox
    rays_filtered, filter_mask  = tensoIR.filtering_rays(all_rays, bbox_only=True)
    rgbs_filtered = all_rgbs[filter_mask, :]                # [filtered(N*H*W), 3]
    light_idx_filtered = all_light_idx[filter_mask, :]      # [filtered(N*H*W), 1]
    # Perf: move filtered tensors to GPU once so per-iteration indexing runs on GPU
    # (avoids CPU gather + PCIe transfer on every one of the 80k training steps)
    rays_filtered      = rays_filtered.to(device)
    rgbs_filtered      = rgbs_filtered.to(device)
    light_idx_filtered = light_idx_filtered.to(device)
    trainingSampler = SimpleSampler(rays_filtered.shape[0], args.batch_size)

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout) if (
            (not is_distributed) or (dist.get_rank() == 0)) else range(args.n_iters)

    # Stage 2: start with relight_flag=True but gate rendering losses behind warmup
    # During warmup, only normal/smoothness losses run so BRDF+Normal MLPs stabilise
    # before the full rgb_brdf + S3IM objective fires.
    # Unified curriculum overrides both: relight_flag becomes per-iteration.
    relight_flag = (args.stage == 2) and not args.unified_curriculum
    stage2_warmup_iters = args.stage2_warmup_iters if (args.stage == 2 and not args.unified_curriculum) else 0
    for iteration in pbar: 
        # Sample batch_size chunk from all rays
        rays_idx = trainingSampler.nextids()
        rays_train = rays_filtered[rays_idx]
        rgb_train = rgbs_filtered[rays_idx]
        light_idx_train = light_idx_filtered[rays_idx]
        rgb_with_brdf_train = rgb_train


        # -------------------------------------------------------------------
        # Unified curriculum: per-iteration schedule factors (before renderer call)
        # relight_flag must be set here so the renderer knows whether to run PBR path.
        # -------------------------------------------------------------------
        if args.unified_curriculum:
            # Phase A (0–nep_ramp_start): NeP off, S3IM at half-weight, no BRDF
            # Phase B (nep_ramp_start–nep_ramp_end): NeP ramps in, no BRDF yet
            # Phase C (brdf_activation_iter – +brdf_warmup_iters): BRDF + S3IM-brdf ramp in
            # Phase D (onwards): full joint
            alpha_nep    = sigmoid_ramp(iteration, args.nep_ramp_start, args.nep_ramp_end)
            brdf_ramp    = smoothstep(iteration, args.brdf_activation_iter,
                                      args.brdf_activation_iter + args.brdf_warmup_iters)
            s3im_rgb_w   = args.s3im_weight * (0.5 + 0.5 * smoothstep(iteration, 0, args.s3im_rgb_ramp_end))
            s3im_brdf_w  = args.s3im_weight * smoothstep(iteration, args.s3im_brdf_ramp_start, args.s3im_brdf_ramp_end)
            relight_flag = (iteration >= args.brdf_activation_iter)
        else:
            alpha_nep    = 1.0
            brdf_ramp    = 1.0
            s3im_rgb_w   = args.s3im_weight
            s3im_brdf_w  = args.s3im_weight

        ret_kw = renderer(
                            rays=rays_train,    # [batch_size, 6]
                            normal_gt=None,     # [batch_size, 3]
                            light_idx=light_idx_train, # [batch_size, 1]
                            tensoIR=tensoIR,    # nn.Module
                            N_samples=nSamples, # int
                            white_bg=white_bg,  # bool
                            ndc_ray=ndc_ray,
                            device=device,
                            sample_method=args.light_sample_train,
                            chunk_size=args.relight_chunk_size,
                            is_train=True,
                            is_relight=relight_flag,
                            args=args
                         )

        total_loss = 0
        loss_rgb_brdf = torch.tensor(1e-6).to(device)
        loss_s3im_brdf = None

        # NeP Eq. 4: Dynamic weight w_s for specular-aware MSE
        # In unified curriculum, alpha_nep interpolates from vanilla MSE (0) to full NeP (1).
        if args.nep_dynamic_weight:
            c_o_map = ret_kw['rgb_map']
            c_a_map = ret_kw['pseudo_albedo_map']
            color_diff = torch.mean((c_o_map - c_a_map) ** 2, dim=-1, keepdim=True)
            w_s_raw = 1.0 / (color_diff + args.nep_epsilon)
            w_s_raw = torch.clamp(w_s_raw, max=args.nep_clamp_max)
            w_s_raw = w_s_raw.detach()  # CRITICAL: stop gradient through weights!
            # Blend: at alpha_nep=0 → uniform weight 1.0 (vanilla MSE); at 1 → full NeP w_s
            w_s = (1.0 - alpha_nep) * torch.ones_like(w_s_raw) + alpha_nep * w_s_raw
            loss_rgb = torch.mean(w_s * (c_o_map - rgb_train) ** 2)
        else:
            w_s = None
            loss_rgb = torch.mean((ret_kw['rgb_map'] - rgb_train) ** 2)

        total_loss += loss_rgb

        # S3IM loss for base geometry/color
        # In unified curriculum s3im_rgb_w ramps from 0.5x→1.0x over the first s3im_rgb_ramp_end iters.
        if s3im_loss_fn is not None:
            loss_s3im = s3im_rgb_w * s3im_loss_fn(ret_kw['rgb_map'], rgb_train)
            total_loss += loss_s3im

        if Ortho_reg_weight > 0:
            loss_reg = tensoIR.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        # v5: skip density regularisation after freeze iter (frozen params, wasted compute)
        _density_frozen = args.unified_curriculum and args.freeze_density_iter > 0 \
                          and iteration >= args.freeze_density_iter
        if L1_reg_weight > 0 and not _density_frozen:
            loss_reg_L1 = tensoIR.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density > 0 and not _density_frozen:
            TV_weight_density *= lr_factor
            loss_tv = tensoIR.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensoIR.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        if relight_flag:
            loss_rgb_brdf = torch.mean((ret_kw['rgb_with_brdf_map'] - rgb_with_brdf_train) ** 2)

            # During Stage 2 warmup (legacy 2-stage): skip rendering and S3IM losses so
            # BRDF/Normal MLPs stabilise on smoothness/normal losses first.
            # In unified curriculum, brdf_ramp (0→1 over brdf_warmup_iters) handles this.
            in_warmup = (stage2_warmup_iters > 0 and iteration < stage2_warmup_iters)
            if not in_warmup:
                total_loss += loss_rgb_brdf * args.rgb_brdf_weight * brdf_ramp

                # Apply S3IM loss to the final PBR color.
                # In unified curriculum, s3im_brdf_w ramps from 0→s3im_weight over Phase C.
                if s3im_loss_fn is not None:
                    loss_s3im_brdf = s3im_brdf_w * s3im_loss_fn(ret_kw['rgb_with_brdf_map'], rgb_with_brdf_train)
                    total_loss += loss_s3im_brdf

            # Exponential enhance anchored to brdf_activation_iter (unified) or first AlphaMask update (legacy).
            enhance_anchor = args.brdf_activation_iter if args.unified_curriculum else update_AlphaMask_list[0]
            enhance_denom  = max(args.n_iters - enhance_anchor, 1)
            t_enhance      = max(iteration - enhance_anchor, 0) / enhance_denom
            normal_weight_factor = args.normals_loss_enhance_ratio ** t_enhance
            BRDF_weight_factor   = args.BRDF_loss_enhance_ratio   ** t_enhance
 
            if args.normals_diff_weight > 0:
                loss_normals_diff = normal_weight_factor * args.normals_diff_weight * ret_kw['normals_diff_map'].mean()
                total_loss += loss_normals_diff
                summary_writer.add_scalar('train/normals_diff_loss', loss_normals_diff.detach().item(), iteration)

            if args.normals_orientation_weight > 0:
                loss_normals_orientation = normal_weight_factor * args.normals_orientation_weight * ret_kw['normals_orientation_loss_map'].mean()
                total_loss += loss_normals_orientation
                summary_writer.add_scalar('train/normals_orientation_loss', loss_normals_orientation.detach().item(), iteration)

            if args.roughness_smoothness_loss_weight > 0: 
                roughness_smoothness_loss = BRDF_weight_factor * args.roughness_smoothness_loss_weight * ret_kw['roughness_smoothness_loss']
                total_loss += roughness_smoothness_loss
                summary_writer.add_scalar('train/roughness_smoothness_loss', roughness_smoothness_loss.detach().item(), iteration)
            
            if args.albedo_smoothness_loss_weight > 0: 
                albedo_smoothness_loss = BRDF_weight_factor * args.albedo_smoothness_loss_weight * ret_kw['albedo_smoothness_loss']
                total_loss += albedo_smoothness_loss
                summary_writer.add_scalar('train/albedo_smoothness_loss', albedo_smoothness_loss.detach().item(), iteration)


        optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        else:
            print(f"[Warning] Iteration {iteration}: total_loss does not require grad. Check frozen parameters.")

        total_loss = total_loss.detach().item()
        loss_rgb = loss_rgb.detach().item()
        loss_rgb_brdf = loss_rgb_brdf.detach().item()

        PSNRs_rgb.append(-10.0 * np.log(loss_rgb) / np.log(10.0))
        if relight_flag:
            PSNRs_rgb_brdf.append(-10.0 * np.log(loss_rgb_brdf) / np.log(10.0))
        else:
            PSNRs_rgb_brdf.append(0.0)
        if (not is_distributed) or (dist.get_rank() == 0):
            summary_writer.add_scalar('train/mse', total_loss, global_step=iteration)
            summary_writer.add_scalar('train/PSNRs_rgb', PSNRs_rgb[-1], global_step=iteration)
            summary_writer.add_scalar('train/mse_rgb', loss_rgb, global_step=iteration)
            if s3im_loss_fn is not None:
                summary_writer.add_scalar('train/s3im_loss', loss_s3im.detach().item(), global_step=iteration)
            if args.nep_dynamic_weight and w_s is not None:
                summary_writer.add_scalar('train/nep_w_s_mean', w_s.mean().item(), global_step=iteration)
            if args.unified_curriculum:
                summary_writer.add_scalar('train/curriculum_alpha_nep',    alpha_nep,    global_step=iteration)
                summary_writer.add_scalar('train/curriculum_brdf_ramp',    brdf_ramp,    global_step=iteration)
                summary_writer.add_scalar('train/curriculum_s3im_brdf_w',  s3im_brdf_w,  global_step=iteration)
                summary_writer.add_scalar('train/curriculum_s3im_rgb_w',   s3im_rgb_w,   global_step=iteration)
            if relight_flag:
                summary_writer.add_scalar('train/PSNRs_rgb_brdf', PSNRs_rgb_brdf[-1], global_step=iteration)
                summary_writer.add_scalar('train/mse_rgb_brdf', loss_rgb_brdf, global_step=iteration)
                if s3im_loss_fn is not None and not in_warmup and loss_s3im_brdf is not None:
                    summary_writer.add_scalar('train/s3im_loss_brdf', loss_s3im_brdf.detach().item(), global_step=iteration)

            # Print the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d} PSNR:'
                    + f' train_rgb = {float(np.mean(PSNRs_rgb)):.2f}'
                    + f' train_rgb_brdf = {float(np.mean(PSNRs_rgb_brdf)):.2f}'
                    + f' test_rgb = {float(np.mean(PSNRs_test)):.2f}'
                    + f' test_rgb_brdf = {float(np.mean(PSNRs_rgb_brdf_test)):.2f}'
                    + f' mse = {float(total_loss):.6f}'
                )
                PSNRs_rgb = []
                PSNRs_rgb_brdf = []

            # Evaluate on testing dataset
            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0 and relight_flag:
                PSNRs_test, PSNRs_rgb_brdf_test, MAE_test, \
                    PSNR_albedo_single, PSNR_albedo_three \
                                                        = evaluation_iter_TensoIR(  
                                                            test_dataset, 
                                                            tensoIR, 
                                                            args, 
                                                            renderer, 
                                                            f'{logfolder}/imgs_vis/',
                                                            prtx=f'{iteration:06d}_', 
                                                            N_samples=nSamples,
                                                            white_bg=white_bg, 
                                                            ndc_ray=ndc_ray,
                                                            compute_extra_metrics=False, 
                                                            logger=summary_writer,
                                                            step=iteration, 
                                                            device=device,
                                                        )
                summary_writer.add_scalar('test/psnr_rgb', np.mean(PSNRs_test), global_step=iteration)
                summary_writer.add_scalar('test/psnr_rgb_brdf', np.mean(PSNRs_rgb_brdf_test), global_step=iteration)
                summary_writer.add_scalar('test/mae', MAE_test, global_step=iteration)
                summary_writer.add_scalar('test/psnr_albedo_single', PSNR_albedo_single, global_step=iteration)
                summary_writer.add_scalar('test/psnr_albedo_three', PSNR_albedo_three, global_step=iteration)

            # Save iteration models
            if iteration % args.save_iters == 0:
                tensoIR.save(f'{logfolder}/checkpoints/{args.expname}_{iteration}.th')


        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # v5 Phase E: freeze density grid to stabilise derived_normals as a fixed target for Normal MLP
        if args.unified_curriculum and args.freeze_density_iter > 0 \
                and iteration == args.freeze_density_iter:
            optimizer.param_groups[0]['lr'] = 0.0   # density_line
            optimizer.param_groups[1]['lr'] = 0.0   # density_plane
            print(f"[v5] Density grid frozen at iter {iteration} — derived_normals now a fixed supervision target.")

        if iteration in update_AlphaMask_list:

            reso_mask = reso_cur  # default: use current resolution
            if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256 ** 3:  # update volume resolution
                reso_mask = reso_cur
            new_aabb = tensoIR.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensoIR.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)
                # In legacy 2-stage: activate relighting after first AlphaMask update.
                # In unified curriculum: relight_flag is determined per-iteration by brdf_activation_iter.
                if not args.unified_curriculum:
                    relight_flag = True
                torch.cuda.empty_cache()
                TV_weight_density = 0
                TV_weight_app = 0


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # Filter rays outside the bbox
                rays_filtered, filter_mask = tensoIR.filtering_rays(all_rays, bbox_only=True)
                rgbs_filtered = all_rgbs[filter_mask, :]                # [filtered(N*H*W), 3]
                light_idx_filtered = all_light_idx[filter_mask, :]      # [filtered(N*H*W), 1]
                rays_filtered      = rays_filtered.to(device)
                rgbs_filtered      = rgbs_filtered.to(device)
                light_idx_filtered = light_idx_filtered.to(device)
                trainingSampler = SimpleSampler(rays_filtered.shape[0], args.batch_size)
    

        if iteration in upsamp_list and args.stage != 2:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensoIR.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensoIR.upsample_volume_grid(reso_cur)

            # Re-apply Stage 2 freezing after upsampling (upsampling creates new nn.Parameters)
            if args.stage == 2:
                for param in tensoIR.density_line.parameters():
                    param.requires_grad = False
                for param in tensoIR.density_plane.parameters():
                    param.requires_grad = False
                for param in tensoIR.app_line.parameters():
                    param.requires_grad = False
                for param in tensoIR.app_plane.parameters():
                    param.requires_grad = False

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensoIR.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    tensoIR.save(f'{logfolder}/{args.expname}.th')

    # if args.render_train:
    #     os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    #     train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    #     PSNRs_test = evaluation(train_dataset, tensoIR, args, renderer, visibility_net, f'{logfolder}/imgs_train_all/',
    #                             N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    #     print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test, PSNRs_rgb_brdf_test, MAE_test,\
            PSNR_albedo_single, PSNR_albedo_three = evaluation_iter_TensoIR(
                                                            test_dataset, 
                                                            tensoIR, 
                                                            args, 
                                                            renderer, 
                                                            f'{logfolder}/imgs_test_all/', 
                                                            N_samples=-1,
                                                            white_bg=white_bg, 
                                                            ndc_ray=ndc_ray, 
                                                            device=device,
                                                            test_all=True
                                                            )
        summary_writer.add_scalar('test/psnr_rgb_all', np.mean(PSNRs_test), global_step=iteration)
        summary_writer.add_scalar('test/psnr_rgb_brdf_all', np.mean(PSNRs_rgb_brdf_test), global_step=iteration)
        summary_writer.add_scalar('test/mae_all', MAE_test, global_step=iteration)
        summary_writer.add_scalar('test/psnr_albedo_single_all', PSNR_albedo_single, global_step=iteration)
        summary_writer.add_scalar('test/psnr_albedo_three_all', PSNR_albedo_three, global_step=iteration)
        print(f'======> {args.expname} test all: nvs psnr: {np.mean(PSNRs_test)}, nvs with brdf psnr: {np.mean(PSNRs_rgb_brdf_test)}, MAE: {MAE_test}  <========================')

    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     # c2ws = test_dataset.poses
    #     print('========>', c2ws.shape)
    #     os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset, tensoIR, c2ws, renderer, visibility_net,  f'{logfolder}/imgs_path_all/',
    #                     N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)
    random.seed(20211202)

    os.environ['PYTHONHASHSEED'] = str(20211202)


    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
