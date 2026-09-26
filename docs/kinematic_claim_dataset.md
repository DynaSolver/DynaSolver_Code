# WaterLily20 kinematic claim (paper / visualization baseline)

Surface+flow, rollout, and main result figures in the paper use this pack.

## Base (full kinematic claim)

- **Manifest:** `waterlily20_grid18x18x36_n11664_f30_kinematic_temporal_manifest.json`
- **Scale:** 348 trajectories = train 216 + test 132 (val 0)

### Training robots (18)

12 train + 4 in-domain test per robot:

`a1`, `anymal_b`, `anymal_d`, `b1`, `b2`, `bolt`, `booster_t1`, `cassie`, `ergocub`, `g1`, `go1`, `go2`, `h1`, `jvrc`, `laikago`, `mini_cheetah`, `spot`, `talos`

### Pure OOD test robots (4)

| robot   | test |
|---------|------|
| aliengo | 16   |
| h1_2    | 16   |
| legolas | 16   |
| solo    | 12   |

### Wind conditions

- **Train winds (12):** `u2_m20`, `u2_p0`, `u3_m10`, `u3_p10`, `u4_m20`, `u4_p20`, `u6_m20`, `u6_p20`, `u7_m10`, `u7_p10`, `u8_p0`, `u8_p20`
- **Test winds (4):** `u2_p20`, `u4_p0`, `u6_p0`, `u8_m20`

Place the manifest under `$DYNASOLVER_DATA/manifests/` and point configs or CLI
flags at it. Visualization tooling beyond train/eval is out of scope for this
anonymous code package.
