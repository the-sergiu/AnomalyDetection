# Avenue

## RBDC_TBDC_AUC on Object Detectors


### After bug fix Experiments

### UNet (`unet_yolov3_norm_0.5_58.pth`) with YoloV3 preds
```py
tbdc = 0.29263123151497894
rbdc = 0.46191602219848915
Macro AUC: 0.787
Micro AUC: 0.728
```

### **UNet (`unet_yolov8_norm_0.5_epoch_30_loss_0.00021063288704248234.pth`) on YoloV8 preds**
```py
tbdc = 0.30934999600906915
rbdc = 0.5402115364172038
Macro AUC:0.7956319678662274 
Micro AUC:0.7545247553060352 # Cred ca s-ar fi obtinut 0.79 daca schimbam randul cu standardizarea
```

### UNet (`unet_detr_norm_0.5_epoch_15_loss_0.0003237094696664687.pth`) with DETR preds
```py
tbdc = 0.32006618473479725
rbdc = 0.4594607941800659
Macro AUC: 0.7860107686095413
Micro AUC: 0.78358873677470787
```


### UNet (`unet_detr_norm_0.5_epoch_15_loss_0.0003237094696664687.pth`)
```py
tbdc = 0.32006618473479725
rbdc = 0.4594607941800659
Macro AUC: 0.7860107686095413
Micro AUC: 0.78358873677470787
```

### AE: `autoencoder_ft_t8_decoder_v2_trained_detrdc5_avenue_loss_0.003.pth` (trained on DETR preds) pred on YoloV3 preds``
```py
tbdc = 0.287895506643282
rbdc = 0.190240156846488
Macro AUC: 0.7140459447941
Micro AUC: 0.68037405196159
```

### AE: `autoencoder_ft_t8_decoder_v2_yolov8_new_epoch_20_mseloss_0.007056034170091152.pth` (trained on YoloV8 preds) on YoloV8 obj preds
```py
tbdc = 0.2552006143876294
rbdc = 0.3212518808549
Macro AUC: 0.7147596333466:
Micro AUC: 0.6468671527616
```

### **AE: `autoencoder_t8_decoder_v2_yolov8_adamw_numheads_16_ffdim_1024_epoch_30_mseloss_0.0016674927901476622.pth`**
```py
tbdc = 0.3085140577843646
rbdc = 0.5458579606408576
Macro AU0.7948291314296569566
Micro AU0.7979429679146359416
```


### AE: `autoencoder_t8_decoder_v2_yolov8_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_29_mseloss_0.001047978294081986.pth`
```py
tbdc = 0.30905445139801235
rbdc = 0.5498900299269054
Macro AUC:0.790 
Micro AUC:0.795 
```


### AE: `autoencoder_t8_decoder_v2_yolov8_adamw_numheads_16_ffdim_2048_epoch_30_mseloss_0.0011713448911905289.pth`
```py
tbdc = 0.307781128721543
rbdc = 0.5487378434034695
Macro AUC:0.7937 
Micro AUC:0.7948 
```

### AE: `autoencoder_t8_decoder_yolov8_adamw_numheads_16_ffdim_1024_numblocks_2`
- autoencoder_t8_decoder_yolov8_adamw_numheads_16_ffdim_1024_numblocks_2_epoch_29_mseloss_0.002231613965705037.pth (30 epochs)
```py
tbdc = 0.31274929835120624
rbdc = 0.5387933196190025
Macro AUC: 0.8010095657345718
Micro AUC: 0.8020006853071939
```
- 'autoencoder_t8_decoder_yolov8_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_76_mseloss_0.0016273667570203543.pth'
```py
tbdc = 0.3120088188786131
rbdc = 0.5382929039341936
Macro AUC: 0.80106
Micro AUC: 0.80133
```

### AE: `autoencoder_t8_decoder_v3_yolov8_adamw_numheads_16_ffdim_1024_numblocks_1_epoch_24_mseloss_0.02316322922706604.pth` (with positional encoding)
24 epochs
```py
tbdc = 0.3140582158165984
rbdc = 0.5397783285562593
Macro AUC: 0.7840071255330223
Micro AUC: 0.7884597852859114
```

### AE: `autoencoder_t12_decoder_v3_yolov8_adamw_numheads_16_ffdim_1024_numblocks_1_epoch_34_mseloss_0.003686577547341585.pth`
```py
tbdc = 0.3104901078845693
rbdc = 0.5408435000380806
Macro AUC: 0.8026187496709234
Micro AUC: 0.797566231784833
```

### AE: `autoencoder_t12_decoder_yolov8_adamw_numheads_32_ffdim_2048_numblocks_2_epoch_80_mseloss_0.0010380609892308712.pth`
```py
tbdc = 0.3110736466969114
rbdc = 0.5466466146930609
Macro AUC: 0.8032072277939423
Micro AUC: 0.8050513428529235
```

### AE **: used `autoencoder_t12_decoder_yolov8_adamw_numheads_32_ffdim_2048_numblocks_2_epoch_80_mseloss_0.0010380609892308712` on YOLOv8_conf_0.25 predictions.
```py
tbdc = 0.24970607333389422
rbdc = 0.298551375454549
Macro AUC: 0.8059229684865824
Micro AUC: 0.7317234053939647
```

### AE: `autoencoder_t12_decoder_DETR101DC5_adamw_numheads_16_ffdim_1024_numblocks_1_epoch_40_mseloss_0.0.pth`
```py
tbdc = 0.3241606640908983
rbdc = 0.4467319891368415
Macro AUC: 0.7590044788836883
Micro AUC: 0.7847173877503629
```

### AE: `autoencoder_t12_decoder_yolov3_adamw_numheads_16_ffdim_1024_numblocks_1_epoch_25_mseloss_0.001866272778196227.pth`
```py
tbdc = 0.3018874945798844
rbdc = 0.46071683521475226
Macro AUC: 0.7796915570336345
Micro AUC: 0.7360502619400648
```


**Similar scores for `autoencoder_t12_decoder_yolov8_2_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_17_mseloss_0.0025767571664901574.pth`**.

### AE: `autoencoder_t12_decoder_yolov8_2_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_17_mseloss_0.0025767571664901574.pth`
- conf-025 (as above) -> poor
- conf 0.35 
```py
tbdc = 0.2446543098817606
rbdc = 0.3222847905776865
Macro AUC: 0.8033719020430768
Micro AUC: 0.7209629854890344

```

- conf 0.5
```
tbdc = 0.2361115605402642
rbdc = 0.3596554584490589
Macro AUC: 0.782983971039827
Micro AUC: 0.7344672750015072

```

- conf 0.7 -> old best
```
tbdc = 0.3110736466969114
rbdc = 0.5466466146930609
Macro AUC: 0.8032072277939423
Micro AUC: 0.8050513428529235
```


- conf 0.77
```
tbdc = 0.32344391447758714
rbdc = 0.593958905291755
Macro AUC: 0.8101914107496304
Micro AUC: 0.8048126974092469
```

- conf 0.8 -> new best
```
tbdc = 0.3401006577485541
rbdc = 0.6184287387657396
Macro AUC: 0.8162230814683262
Micro AUC: 0.8271392844099114
```

- conf 0.83
```
tbdc = 0.36928838466533354
rbdc = 0.6459050511339665
Macro AUC: 0.8079421588784405
Micro AUC: 0.8417824722291121
```

- conf 0.84
```
tbdc = 0.3694825380594584
rbdc = 0.6490911454800796
Macro AUC: 0.8131569223061648
Micro AUC: 0.847790049069593
```


- **conf 0.85**
```
tbdc = 0.3600634018694815
rbdc = 0.6433398830533846
Macro AUC: 0.8265188476242953
Micro AUC: 0.8501921904999612
```

- **conf 0.86**
```
tbdc = 0.342297287677084
rbdc = 0.6335809435625008
Macro AUC: 0.8306437854564369
Micro AUC: 0.8600054545595308
```


- conf 0.9 
```
tbdc = 0.30776602790200003
rbdc = 0.47020455920804205
Macro AUC: 0.8153305223700114
Micro AUC: 0.8446444569985535
```

-----------------------------------------
## RBDC_TBDC_AUC on Object Detectors + Optical Flow (merged)

## AE

### `autoencoder_t12_decoder_yolov8_2_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_17_mseloss_0.0025767571664901574.pth`
- Chosen model: YoloV8
- Object Detection on 0.85 conf:
```
tbdc = 0.3600634018694815
rbdc = 0.6433398830533846
Macro AUC: 0.8265188476242953
Micro AUC: 0.8501921904999612
```
- RAFT for Optical Flow
### Experiment 1
- pixel threshold: 200
- 200 min area
- **IoU: 0.5**
```
tbdc = 0.4999886743853427
rbdc = 0.6193564768334915
Macro AUC: 0.8330236334073822
Micro AUC: 0.8636205955915847
```

### Experiment 2
- pixel threshold: 200
- 200 min area
- IoU: 0.3.
```
tbdc = 0.5121464520624484
rbdc = 0.6221676012072133 
Macro AUC: 0.8356354333805757
Micro AUC: 0.8589531168577872
```


### Experiment 3
- pixel threshold: 200
- 200 min area
- IoU: 0.4.
```
tbdc = 0.5139914486216188
rbdc = 0.6233028815700441
Macro AUC: 0.8374450689066141
Micro AUC: 0.8591436179023944
```

### Experiment 4
- pixel threshold: 200
- 200 min area
- IoU: 0.6.
```
tbdc = 0.5118848842953634
rbdc = 0.6220041916607746
Macro AUC: 0.8356707420840527
Micro AUC: 0.8589505663106262
```

- Object Detection on 0.86 conf:
```
tbdc = 0.342297287677084
rbdc = 0.6335809435625008
Macro AUC: 0.8306437854564369
Micro AUC: 0.8600054545595308
```
- RAFT for Optical Flow
### Experiment 1 
- pixel threshold: 200
- 200 min area
- IoU: 0.5
```
tbdc = 0.5215828463319031
rbdc = 0.6248973792716732
Macro AUC: 0.8441935326960842
Micro AUC: 0.860101359647035
```

### Experiment 2
- pixel threshold: 200
- 200 min area
- IoU: 0.4
```
tbdc = 0.521417815946897
rbdc = 0.625002417547621
Macro AUC: 0.8441831807966389
Micro AUC: 0.860127609968703
```

### Experiment 3
- pixel threshold: 200
- 200 min area
- IoU: 0.6
```
tbdc = 0.5204497455511907
rbdc = 0.6247327859605374
Macro AUC: 0.8446302801878547
Micro AUC: 0.8609363139885166
```

### Experiment 4
- pixel threshold: 200
- 200 min area
- IoU: 0.7
- no `pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))`
- Range: 224  ; Mu: 21
```
tbdc = 0.5208472206941631
rbdc = 0.6260651047908525
Macro AUC: 0.8461650420761915
Micro AUC: 0.8657940003411414
```

### Experiment 5: **`obj_dect_0.86_avenue_optical_flow_raft_thresh_200_minarea_200_iou_0.8`**
- pixel threshold: 200
- 200 min area
- IoU: 0.8
- no `pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))`
- Range: 214  ; Mu: 21 
```
tbdc = 0.5162829979872765
rbdc = 0.6211594004855369
Macro AUC: 0.8429541855181059
Micro AUC: 0.8667391924909003
```

---------------------
# Shanghaitech

## Object Detector YoloV8

### AE:  `shanghaitech_autoencoder_t12_decoder_yolov8_2_adamw_numheads_8_ffdim_2048_numblocks_1_epoch_17_mseloss_0.0027697434455824254.pth`

### Chosen model: YoloV8
------
### Experiment 1
Object Detection on 0.3 conf: `obj_dect_shanghaitech_yolov8_conf_035`

```py
tbdc = 0.3700040558710984
rbdc = 0.1442493969393207
Macro AUC: 0.756947890588308
Micro AUC: 0.6862264376353493
```

### Experiment 1
Object Detection on 0.5 conf: `obj_dect_shanghaitech_yolov8_conf_05`
- Range: 826; Mu: 249
```py
tbdc = 0.38469813240634804
rbdc = 0.1538486564429625
Macro AUC: 0.8227306825529249
Micro AUC: 0.7486203218121336
```

### Experiment 3: `obj_dect_shanghaitech_yolov8_v8_conf_06_retina`
- Range: 824; Mu: 249 
```py
tbdc = 0.4219972511930995
rbdc = 0.17095409135069528
Macro AUC: 0.821229982250492
Micro AUC: 0.7476275793614825
```

### Experiment 4
Object Detection on 0.7 conf: `obj_dect_shanghaitech_yolov8_conf_07`
```py
tbdc = 0.45057516100132533
rbdc = 0.17743671491699703
Macro AUC: 0.7444712850419362
Micro AUC: 0.6929039185119026
```

### Experiment 5
Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085`
- Range: 822; Mu: 249
```py
tbdc = 0.5228609003785583
rbdc = 0.22173908255269628
Macro AUC: 0.8131065348776778
Micro AUC: 0.7417920506075211
```
-----
## Optical Flow
- Optical Flow RAFT, pixel_threshold = 0, min_area = 200
- conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_200`. Just optical flow:
```py
tbdc = 0.42198344199157856
rbdc = 0.16866178377555585
Macro AUC: 0.7224754662326697
Micro AUC: 0.6758359240268325
```

- conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_100_iterations_50`. Just optical flow:
- Range: 590; Mu: 129
```py
tbdc = 0.41733905065308213
rbdc = 0.15417255188066142
Macro AUC:  0.8141320713802158
Micro AUC:  0.7486532521991696
```

- conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_300_iterations_50`. Just optical flow:
* also enabled test mode on
```py
tbdc = 0.4331006250137704
rbdc = 0.17332374411448986
Macro AUC: 0.724575817073061
Micro AUC: 0.6746765717243532
```

- conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3` Just optical flow:
* also enabled test mode on
- Shitty results - worse tbdc si rbdc (by far) and same macro and micro

- conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_1000_iterations_50`. Just optical flow:
* also enabled test mode on
- didn't get great results either

- TODO: conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- Range: 640; Mu: 108
```py
tbdc = 0.2653551338732266
rbdc = 0.08851432660905945
Macro AUC: 0.8246352794055241
Micro AUC: 0.7543586729644352
```

- TODO: conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_10`

- TODO: conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_15`

- TODO: conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_20`

- TODO: conf name: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_25`
-----
## Object Detector YoloV8 + Optical Flow RAFT
- range: 302; Mu: 25

## YoloV8 CONF: 0.5
### Experiment 1:
- Object Detection on 0.5 conf: `obj_dect_shanghaitech_yolov8_conf_05` + Optical Flow
- File Name: `obj_dect_05_shanghaitech_optical_flow_raft_thresh_200_minarea_100_iou_03`
```py
tbdc = 
rbdc =
Macro AUC: 
Micro AUC: 
```

### Experiment
- Object Detection on 0.5 conf: `obj_dect_shanghaitech_yolov8_conf_05` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_05_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_05`
- Range: 818    ; Mu: 249
```py
tbdc = 0.28963310658314123
rbdc = 0.08602202529107923
Macro AUC: 0.829335505423615  
Micro AUC: 0.7542166403778642
```

### Experiment
- Object Detection on 0.5 conf: `obj_dect_shanghaitech_yolov8_conf_05` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_15`
- File name: `obj_dect_05_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_15_iou_05`
- Range: 816  ; Mu: 249 
```py
tbdc = 0.3284499962606544
rbdc = 0.10818976273473364
Macro AUC:  0.8204916797043775  
Micro AUC:  0.7471267093553169  
```

### Experiment
- Object Detection on 0.5 conf: `obj_dect_shanghaitech_yolov8_conf_05` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_25`
- File name: `obj_dect_05_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_25_iou_05`
- Range: 814  ; Mu: 249
```py
tbdc = 0.316580909973626
rbdc = 0.1016178024074119
Macro AUC:  0.8236589592232842  
Micro AUC:  0.7498274364581642  
```


## YoloV8 CONF: 0.6
### Experiment
- Object Detection on 0.6 conf: `obj_dect_shanghaitech_yolov8_conf_06_retina` + Optical Flow
- File name: `obj_dect_06_shanghaitech_optical_flow_raft_thresh_0_minarea_100_lookahead_3_iou_03`
- Range: 408; Mu: 86
```py
tbdc = 0.4270160425683377
rbdc = 0.15614327352529173
Macro AUC: 0.8119450372885829
Micro AUC: 0.7500978134326315
```

### Experiment
- Object Detection on 0.6 conf: `obj_dect_shanghaitech_yolov8_conf_06_retina` + Optical Flow
- File name: `obj_dect_06_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_500_lookahead_3_iou_03`
- Range: 842 ; Mu: 249
```py
tbdc = 0.44004820497447783
rbdc = 0.16405193751814198
Macro AUC: 0.820705993390589
Micro AUC: 0.7472958010386748
```

### Experiment
- Object Detection on 0.6 conf: `obj_dect_shanghaitech_yolov8_conf_06_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_06_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_1000_lookahead_3_iou_03`
- Range: 848; Mu: 249 
```py
tbdc = 0.47408819704272065
rbdc = 0.18320269607208894
Macro AUC: 0.8192389903884494
Micro AUC: 0.7494360031699556
```

### Experiment
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_06_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_06_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_03`
- Range: 408  ; Mu: 88
```py
tbdc = 0.31766656146847977
rbdc = 0.10560921209185417
Macro AUC: 0.8204586995237299
Micro AUC: 0.755852130460512  
```

### Experiment
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_06_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_15`
- File name: `obj_dect_06_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_15_iou_03`
- Range: 826  ; Mu: 249
```py
tbdc = 0.2982280156735989
rbdc = 0.09379857485722608
Macro AUC:  0.825349113129101  
Micro AUC:  0.750970180904857
```

### Experiment
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_06_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_25`
- File name: `obj_dect_06_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_25_iou_03`
- Range: 818  ; Mu: 249
```py
tbdc = 0.2904683305358064
rbdc = 0.08602895769634901
Macro AUC:  0.8292875496200671  
Micro AUC:  0.754202509434219  
```

## YoloV8 CONF: 0.7
### Experiment:
- Object Detection on 0.7 conf: `obj_dect_shanghaitech_yolov8_conf_07` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_07_shanghaitech_optical_flow_raft_thresh_0_minarea_300_iou_0.3`
- Range: 822 ; Mu: 249
```py
tbdc = 0.4589399684529567
rbdc = 0.17755066657135032
Macro AUC: 0.8130524679640702
Micro AUC: 0.7417619267316597
```

### Experiment
- Object Detection on 0.7 conf: `obj_dect_shanghaitech_yolov8_conf_07` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_20`
- File name: `obj_dect_07_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_20_iou_03`
- Range: 818 ; Mu: 249
```py
tbdc = 0.31087507203282927
rbdc = 0.08981917745821769
Macro AUC:  0.8210373316473792
Micro AUC:  0.7473519533994192
```

### Experiment
- Object Detection on 0.7 conf: `obj_dect_shanghaitech_yolov8_conf_07` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_10`
- File name: `obj_dect_07_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_10_iou_03`
- Range: 382; Mu: 81
```py
tbdc = 0.325067207366728
rbdc = 0.10599099537904766
Macro AUC: 0.8227038588492961
Micro AUC: 0.7627035498480044
```


## YoloV8 CONF: 0.75

### Experiment
- Object Detection on 0.75 conf: `obj_dect_shanghaitech_yolov8_conf_075_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_075_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_3_iou_05`
- Range: 672 ; Mu: 85
```py
tbdc = 0.38308648997715744
rbdc = 0.12153244988288164
Macro AUC:  0.8189928458306952
Micro AUC:  0.7555109880815759  
```

### TODO: Experiment
- Object Detection on 0.75 conf: `obj_dect_shanghaitech_yolov8_conf_075_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_4`
- File name: `obj_dect_075_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: 392  ; Mu: 73
```py
tbdc = 0.4447681590224202
rbdc = 0.1523216048518976
Macro AUC:  0.8151547136470562  
Micro AUC:   0.7533361165594024
```

### Experiment
- Object Detection on 0.75 conf: `obj_dect_shanghaitech_yolov8_conf_075_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_075_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_05`
- Range: 814; Mu:249 
```py
tbdc = 0.31716927505415843
rbdc = 0.10164951282143306
Macro AUC: 0.8235395614914913 
Micro AUC: 0.7497964885847187  
```

### Experiment
- Object Detection on 0.75 conf: `obj_dect_shanghaitech_yolov8_conf_075_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_7`
- File name: `obj_dect_075_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_7_iou_05`
- Range: 522; Mu: 92
```py
tbdc = 0.3689820197989815
rbdc = 0.1143176258014425
Macro AUC: 0.8168984942654725 
Micro AUC: 0.7506974131075485  
```

### Experiment
- Object Detection on 0.75 conf: `obj_dect_shanghaitech_yolov8_conf_075_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_10`
- File name: `obj_dect_075_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_10_iou_05`
- Range: 666; Mu: 95
```py
tbdc = 0.3151387017923102
rbdc = 0.08662490505469009
Macro AUC:  0.820520956365179
Micro AUC: 0.7505915749223246  
```

### Experiment
- Object Detection on 0.75 conf: `obj_dect_shanghaitech_yolov8_conf_075_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_13`
- File name: `obj_dect_075_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_13_iou_05`
- Range: 526; Mu: 87
```py
tbdc = 0.3483889316612097
rbdc = 0.10486027541699172
Macro AUC:  0.8134353157633739
Micro AUC: 0.7503547211212153 
```

## YoloV8 CONF: 0.78

### TODO: Experiment
- Object Detection on 0.78 conf: `obj_dect_shanghaitech_yolov8_conf_078_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_078_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_3_iou_08`
- Range:684   ; Mu: 76
```py
tbdc = 0.4141651065372139
rbdc = 0.14363004122391365
Macro AUC:  0.8184547102714395  
Micro AUC:  0.7591167646145626 
```

### TODO: Experiment
- Object Detection on 0.78 conf: `obj_dect_shanghaitech_yolov8_conf_078_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_4`
- File name: `obj_dect_078_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: 400  ; Mu: 72
```py
tbdc = 0.40453574860922725
rbdc = 0.14106649212695538
Macro AUC:  0.8204156116705347  
Micro AUC: 0.7589386163368614    
```

### TODO: Experiment
- Object Detection on 0.78 conf: `obj_dect_shanghaitech_yolov8_conf_078_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_078_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_08`
- Range: 382  ; Mu: 71
```py
tbdc = 0.39653680741811687
rbdc = 0.14045300209376632
Macro AUC:  0.820389682739875  
Micro AUC:  0.7602233490956283   
```

### TODO: Experiment
- Object Detection on 0.78 conf: `obj_dect_shanghaitech_yolov8_conf_078_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_7`
- File name: `obj_dect_078_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_7_iou_08`
- Range: 400  ; Mu: 72
```py
tbdc = 0.39171941316168957
rbdc = 0.13734339435964
Macro AUC:  0.8193642534009232  
Micro AUC:   0.7600012780571485  
```

## YoloV8 CONF: 0.8
### Experiment
- Object Detection on 0.8 conf: `obj_dect_shanghaitech_yolov8_conf_08_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_08_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_3_iou_05`
- Range: 698; Mu: 87
```py
tbdc = 0.39739902913555325
rbdc = 0.12759115719057806
Macro AUC: 0.827534172758896
Micro AUC: 0.7650515986057369
```

### TODO: Experiment
- Object Detection on 0.8 conf: `obj_dect_shanghaitech_yolov8_conf_08_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_4`
- File name: `obj_dect_08_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: ; Mu: 
```py
tbdc = 0.39185145395376053
rbdc = 0.13735274760484525
Macro AUC: 0.8193897741668632
Micro AUC: 0.7599406318349522
```

### Experiment
- Object Detection on 0.8 conf: `obj_dect_shanghaitech_yolov8_conf_08_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_08_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_05`
- Range: 498; Mu: 79 
```py
tbdc = 0.37700175237215705
rbdc = 0.12271476221603311
Macro AUC:  0.8265833752024346
Micro AUC:  0.766351064933248  
```

### Experiment
- Object Detection on 0.8 conf: `obj_dect_shanghaitech_yolov8_conf_08_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_7`
- File name: `obj_dect_08_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_7_iou_05`
- Range: 398; Mu: 81 
```py
tbdc = 0.36596633844211723
rbdc = 0.11779423804805465
Macro AUC:  0.8245618406131767
Micro AUC: 0.7606649650665609  
```

### Experiment
- Object Detection on 0.8 conf: `obj_dect_shanghaitech_yolov8_conf_08_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_10`
- File name: `obj_dect_08_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_10_iou_05`
- Range: 374; Mu: 92 
```py
tbdc = 0.3720163203039638
rbdc = 0.11351334639543478
Macro AUC: 0.8242260028790831 
Micro AUC: 0.7583567633561831  
```

### Experiment
- Object Detection on 0.8 conf: `obj_dect_shanghaitech_yolov8_conf_08_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_13`
- File name: `obj_dect_08_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_13_iou_05`
- Range: 504; Mu: 78 
```py
tbdc = 0.3489852098796924
rbdc = 0.10484182266167603
Macro AUC: 0.8211563722859836 
Micro AUC: 0.7581735284307084  
```

## YoloV8 CONF: 0.82
### TODO: Experiment
- Object Detection on 0.82 conf: `obj_dect_shanghaitech_yolov8_v8_conf_082_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_082_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_3_iou_08`
- Range: 406  ; Mu: 79
```py
tbdc = 0.4134448558241781
rbdc = 0.1425660212276308
Macro AUC: 0.8224556337702029
Micro AUC: 0.7652542528175613
```

### TODO: Experiment
- Object Detection on 0.82 conf: `obj_dect_shanghaitech_yolov8_v8_conf_082_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_4`
- File name: `obj_dect_082_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: 370  ; Mu: 75
```py
tbdc = 0.4109100139674643
rbdc = 0.14035638339443773
Macro AUC: 0.8266273761506927
Micro AUC: 0.7666875068364128
```

### Experiment
- Object Detection on 0.82 conf: `obj_dect_shanghaitech_yolov8_v8_conf_082_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_082_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_08`
- Range: 378; Mu: 73
```py
tbdc = 0.40268345369061
rbdc = 0.13974631878502908
Macro AUC: 0.8257309040671714
Micro AUC: 0.7692469072849478 
```

### TODO: Experiment
- Object Detection on 0.82 conf: `obj_dect_shanghaitech_yolov8_v8_conf_082_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_7`
- File name: `obj_dect_082_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_7_iou_08`
- Range: 402 ; Mu: 77
```py
tbdc = 0.39218473670507487
rbdc = 0.13478514651377033
Macro AUC: 0.8264929804385175
Micro AUC: 0.7635603868105387
```

### TODO: Experiment
- Object Detection on 0.82 conf: `obj_dect_shanghaitech_yolov8_v8_conf_082_retina` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_10`
- File name: `obj_dect_082_retina_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_10_iou_08`
- Range: ; Mu: 
```py
tbdc = 
rbdc = 
Macro AUC: 
Micro AUC: 
```


## YoloV8 CONF: 0.85
### Experiment:
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_100_lookahead_3_iou_03`
- Range: 600; Mu: 96 
```py
tbdc = 0.4460528802425703
rbdc = 0.16450869952586117
Macro AUC: 0.8205674428792512
Micro AUC: 0.7572192408404491
```

### Experiment
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_3`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_3_iou_03`
- Range: 558; Mu: 94
```py
tbdc = 0.34190900263740237
rbdc = 0.11197052827518585
Macro AUC:  0.8213742386794874
Micro AUC:  0.7602897952769394
```

### Experiment
Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_30`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_30_iou_03`
- Range: 816; Mu: 201
```py
tbdc = 0.25315121323920736
rbdc = 0.06791684951290118
Macro AUC: 0.8257891804626449
Micro AUC: 0.7511087281046316
```


### Experiment
Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_25`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_25_iou_03`
- Range: 810; Mu: 201
```py
tbdc = 0.2591670599961707
rbdc = 0.06859996782198845
Macro AUC:  0.8283608200458892
Micro AUC:  0.7528833212779618
```

### Experiment
Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_20`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_20_iou_03`
- Range: 810; Mu: 249
```py
tbdc = 0.2759325167595607
rbdc = 0.07687532174758954
Macro AUC:  0.8240351898310947
Micro AUC:  0.7492485646269544
```

### Experiment
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_15`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_15_iou_03`
- Range: 618; Mu: 109
```py
tbdc = 0.28817802015771044
rbdc = 0.08284889057443212
Macro AUC:  0.8220375473372966
Micro AUC: 0.7508257550357504
```

### Experiment
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_10`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_10_iou_03`
- Range: ; Mu: 
```py
tbdc = 0.31022076413604516
rbdc = 0.09520154610321298
Macro AUC: 0.8282085267930923  
Micro AUC: 0.7549779969277927
```

### TODO: Experiment
- Object Detection on 0.85 conf: `obj_dect_shanghaitech_yolov8_conf_085` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_085_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_03`
- Range: ; Mu: 
```py
tbdc = 0.32625619513363735
rbdc = 0.10593141423616506
Macro AUC: 0.822919981717193
Micro AUC: 0.7626745082376268
```

## DETR 50 DC5 Object Detector + Optical Flow
### TODO: Experiment
- Object Detection using DETR 50 DC5, 0.95 threshold : `obj_dect_shanghaitech_detr_resnet_50_dc5_thresh_95` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_4`
- File name: `obj_dect_095_detr50dc5_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: 594 ; Mu:112
```py
tbdc = 0.2779742615102798
rbdc = 0.0925387438634669
Macro AUC: 0.824651230289233  
Micro AUC: 0.7534795462523249
```

### TODO: Experiment
- Object Detection using DETR 50 DC5, 0.95 threshold : `obj_dect_shanghaitech_detr_resnet_50_dc5_thresh_95` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_5`
- File name: `obj_dect_095_detr50dc5_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_5_iou_08`
- Range: 640  ; Mu: 108
```py
tbdc = 0.26544760897554653
rbdc = 0.08847547924821247
Macro AUC: 0.8245645660438191
Micro AUC: 0.7543272060954963
```

### Experiment
- Object Detection using DETR 50 DC5, 0.92 threshold : `obj_dect_shanghaitech_detr_resnet_50_dc5_thresh_92` + 
- Optical Flow conf: `optical_flow_raft_shanghaitech_thresh_0_minarea_0_iterations_50_lookahead_4`
- File name: `obj_dect_092_detr_shanghaitech_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: ; Mu: 
```py
tbdc = 0.2764914015222395
rbdc = 0.09249689569675337
Macro AUC: 0.8247079538141868 
Micro AUC: 0.753486974529054
```

-----------
# UBNormal

## Object Prediction YoloV8

### YoloV8 Conf 0.65
- Object Detection 0.65 threshold : `obj_dect_ubnormal_yolov8_conf_065` + 
- Range: 452; Mu: 201
```py
tbdc = 0.16407346527185676
rbdc = 0.09692325955272213
Macro AUC:  0.7289346254632723
Micro AUC:  0.5594131993798793
```

### YoloV8 Conf 0.75
- Object Detection 0.75 threshold : `obj_dect_ubnormal_yolov8_conf_075` + 
- Range: 452; Mu: 201
```py
tbdc = 0.16444286854244552
rbdc = 0.09398849998183875
Macro AUC: 0.7327446990819347
Micro AUC: 0.564937751904004
```

### YoloV8 Conf 0.80
- Object Detection 0.75 threshold : `obj_dect_ubnormal_yolov8_conf_080` + 
- Range: 452; Mu: 201
```py
tbdc = 0.19272985396034192
rbdc = 0.1039739839474433
Macro AUC:  0.7355763048821048
Micro AUC:  0.5678144603409326
```

### TODO: YoloV8 Conf 0.82?
- Object Detection 0.75 threshold : `obj_dect_ubnormal_yolov8_conf_082` + 
- Range: 452; Mu: 201
```py
tbdc = 
rbdc = 
Macro AUC:  
Micro AUC:  
```

### TODO: YoloV8 Conf 0.83?
- Object Detection 0.75 threshold : `obj_dect_ubnormal_yolov8_conf_083?` + 
- Range: 452; Mu: 201
```py
tbdc = 
rbdc = 
Macro AUC:  
Micro AUC:  
```

### TODO: YoloV8 Conf 0.84?
- Object Detection 0.75 threshold : `obj_dect_ubnormal_yolov8_conf_084?` + 
- Range: 452; Mu: 201
```py
tbdc = 
rbdc = 
Macro AUC:  
Micro AUC:  
```



### YoloV8 Conf 0.85 -> Issues with NaN?
- Object Detection 0.85 threshold : `obj_dect_ubnormal_yolov8_conf_085` + 
- Range: 452; Mu: 201 -> could not perform :
`pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))`
```py
tbdc = 0.24583579283356286
rbdc = 0.12195520640369963
Macro AUC: 0.72 ish
Micro AUC: 0.47252782620077904
```

---
## Optical Flow RAFT
Lookahead: 3, 4, 5, 7, 10, 15, 20, 25
- The longer the look-ahead window, the worse the results.

### RAFT Lookahead 3
- Lookahead 4 : `optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_3` + 
- Range: 452; Mu: 201
```py
tbdc = 0.165057675280383
rbdc = 0.09023906608672382
Macro AUC:  0.7325930218875573
Micro AUC:  0.5560837847502103
```


### RAFT Lookahead 4
- Lookahead 4 : `optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_4` + 
- Range: 452; Mu: 201
```py
tbdc = 0.14994944360639248
rbdc = 0.08789478659124356
Macro AUC: 0.7330077989953476
Micro AUC: 0.5544875108320454
```

### RAFT Lookahead 5
- Lookahead 4 : `optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_5` + 
- Range: 452; Mu: 201
```py
tbdc = 0.1436247021271944
rbdc = 0.08688850849617595
Macro AUC: 0.7330233981594717
Micro AUC: 0.5505070818977833
```

### RAFT Lookahead 7
- Lookahead 4 : `optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_7` + 
- Range: 452; Mu: 201
```py
tbdc = 
rbdc = 
Macro AUC:  0.7343131695856331
Micro AUC:  0.5515365508963864
```

### RAFT Lookahead 10
- Lookahead 4 : `optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_10` + 
- Range: 452; Mu: 201
```py
tbdc = 
rbdc = 
Macro AUC:  0.7305074077794593
Micro AUC:  0.5460942369411876
```

### RAFT Lookahead 15
- Lookahead 4 : `optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_4` + 
- Range: 452; Mu: 201
```py
tbdc = 
rbdc = 
Macro AUC:  0.7321006654478265
Micro AUC:  0.5453170182147878
```

### RAFT Lookahead 20
- Lookahead 4 : `optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_4` + 
- Range: 452; Mu: 201
```py
tbdc = 
rbdc = 
Macro AUC:  0.7306045309432799
Micro AUC:  0.5449302642864546
```
## Yolov8 + Optical Flow RAFT

### Experiment: `obj_dect_075_yolov8_ubnormal_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: 452; Mu: 201
```py
tbdc = 0.1624114579917361
rbdc = 0.0720157706455589
Macro AUC:  0.7341904606381994
Micro AUC:  0.5614641824194676
```

### Experiment:  `obj_dect_085_yolov8_ubnormal_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: 452; Mu: 201
```
tbdc = 0.15811156839596863
rbdc = 0.07920534759002097
Macro AUC:  0.7360887028923676
Micro AUC:  0.5599474971602325
```

### Experiment: `obj_dect_090_yolov8_ubnormal_optical_flow_raft_thresh_0_minarea_0_lookahead_4_iou_08`
- Range: 452; Mu: 201
```
tbdc = 0.14696456680002623
rbdc = 0.08166837907625453
Macro AUC:  0.7306779744271983
Micro AUC:  0.552354834141613
```

### TODO: `obj_dect_080_yolov8_ubnormal_optical_flow_raft_thresh_0_minarea_0_lookahead_3_iou_08`
- Range: 452; Mu: 201
```py
tbdc = 0.16634119280295578
rbdc = 0.0742444046485038
Macro AUC: 0.7339368826348036
Micro AUC: 0.5594402090449347
```

### TODO: `obj_dect_080_yolov8_ubnormal_optical_flow_raft_thresh_0_minarea_200_lookahead_3_iou_08`
- Range: 452; Mu: 201
```py
tbdc = 0.1660709210555082
rbdc = 0.07421608880186785
Macro AUC: 0.7339740386384328
Micro AUC: 0.5594548025934202
```

### TODO: `obj_dect_080_yolov8_ubnormal_optical_flow_raft_thresh_0_minarea_1000_lookahead_3_iou_08`
- Range: 452; Mu: 201
```py
tbdc = 0.16587190649526684
rbdc = 0.07420974545848795
Macro AUC: 0.7339019008702834
Micro AUC: 0.5594297458211709

```


--------------------------------

# Autoencoder K Means

# Avenue
## Checkpoint: `autoencoder_t12_decoder_yolov8_2_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_17_mseloss_0.0025767571664901574.pth`
- Decoder
```py
input_channels= 512
num_upsamples = 5
num_blocks = 2
num_heads = 8
ff_dim = 1024
output_channels = 3
decoder = UpsampleTransformerDecoder(
    input_channels=input_channels,
    num_upsamples=num_upsamples,  # Adjusted to 5 upsampling steps
    num_blocks=num_blocks,
    num_heads=num_heads,
    ff_dim=ff_dim,
    output_channels=output_channels,
    # use_positional_encoding=True
)

# Set up loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001)
```

### K-Means Cluster Proximity 

### Avenue
#### Experiment 1 - Correct setup, trained autoencoder on cluster loss + reconstruction
- Test time: Recon Loss:
- Checkpoint: `autoencoder_avenue_cluster_and_recon_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_0.002074952055127128.pth`
```py
tbdc = 0.5152734003378269
rbdc = 0.6196390131230487
Macro AUC: 0.8431105627286298
Micro AUC: 0.863995548595496
```

#### Experiment 2 - Correct setup, trained autoencoder on cluster loss + reconstruction
- Test time:  Recon Loss and Cluster Loss (MSE):
- Checkpoint: `autoencoder_avenue_cluster_and_recon_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_0.002074952055127128.pth`
```py
tbdc = 0.5158758151745979
rbdc = 0.615548189093439
Macro AUC: 0.8463287073097415
Micro AUC: 0.8635745954578276
```

#### Experiment 3 - Correct setup, trained autoencoder on cluster loss + reconstruction
- Test time:  Cluster Loss (MSE):
- Checkpoint: `autoencoder_avenue_cluster_and_recon_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_0.002074952055127128.pth`
```py
tbdc = 0.49282603208708425
rbdc = 0.6366112644781091
Macro AUC: 0.8462240909040462
Micro AUC: 0.8640082110464469
```

#### Experiment 4 - Correct setup, trained autoencoder on cluster loss (MSE)
- Test time: Cluster Loss (MSE)
- Checkpoint: `autoencoder_avenue_cluster_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_6.556030938231856e-08.pth`
```py
tbdc = 0.4431729194845875
rbdc = 0.40867635499709426
Macro AUC: 0.7827725080178887
Micro AUC: 0.79
```

#### Experiment 5 - Correct setup, trained autoencoder on cluster loss (MSE)
- Test time: Cluster Loss (MSE) + Reconstruction
- Checkpoint: `autoencoder_avenue_cluster_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_6.556030938231856e-08.pth`
```py
tbdc = 0.5165078923354712
rbdc = 0.6195026634372232
Macro AUC: 0.8440612780526345
Micro AUC: 0.8640082110464469
```

#### Experiment 6 - Correct setup, Fixed Clusters, trained autoencoder on cluster loss (MSE) & Recon loss
- Test time: Cluster Loss (MSE) + Recon loss 
- Checkpoint: `autoencoder_avenue_cluster_and_recon_keep_centroids_fixed_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_0.0030212748409239516.pth`
```py
tbdc = 0.5141861413307274
rbdc = 0.6084547709185226
Macro AUC: 0.8484197658080729
Micro AUC: 0.8643625791033582
```

#### Experiment 7 - Correct setup, Fixed Clusters, trained autoencoder on cluster loss (MSE) & Recon loss
- Test time: Cluster Loss (MSE) 
- Checkpoint: `autoencoder_avenue_cluster_and_recon_keep_centroids_fixed_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_0.0030212748409239516.pth`
```py
tbdc = 0.5433754861924578
rbdc = 0.5064072848145864
Macro AUC: 0.7774770363207649
Micro AUC: 0.8251451628116953
```

#### Experiment 8 - Correct setup, Fixed Clusters, trained autoencoder on cluster loss (MSE) & Recon loss
- Test time: Reconstruction loss (MSE) 
- Checkpoint: `autoencoder_avenue_cluster_and_recon_keep_centroids_fixed_t12_decoder_yolov8_clusters_50_adamw_numheads_8_ffdim_1024_numblocks_2_epoch_20_kmeans_loss_0.0030212748409239516.pth`
```py
tbdc = 0.5143641152753419
rbdc = 0.6084426831804176
Macro AUC: 0.8486186119781884
Micro AUC: 0.8643149764134201
```
-------

#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_loss_0.12655426589204968_abloss_2.3640876997400215.pth
- This approach used a cycle over the abnormal dataloader whilst parsing the entire normal dataloader. Similarly below.
```py
tbdc = 0.4913391406770776
rbdc = 0.5960572365903469
Macro AUC: 0.8290201765860639
Micro AUC: 0.8545735113849994
```

#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_loss_0.0047759712304850585_abloss_0.015505864898149124.pth
```py
tbdc = 0.45273389551527227
rbdc = 0.48775758886538256
Macro AUC: 0.8043156037766543
Micro AUC: 0.8429278484742051
```

```
--- Epoch 1 ---
Epoch 1, Normal Loss: 0.0047759712304850585
Epoch 1, Abnormal Loss: 0.015505864898149124
--- Epoch 2 ---
Epoch 2, Normal Loss: 0.062307134434129906
Epoch 2, Abnormal Loss: 1.110990855656229
--- Epoch 3 ---
Epoch 3, Normal Loss: 0.12655426589204968
Epoch 3, Abnormal Loss: 2.3640876997400215
--- Epoch 4 ---
Epoch 4, Normal Loss: 0.12644549340354658
Epoch 4, Abnormal Loss: 2.3635909006674103
--- Epoch 5 ---
Epoch 5, Normal Loss: 0.12636367545454238
Epoch 5, Abnormal Loss: 2.3636644987903015

Rest of epochs are the same
```
---------
#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_epoch_0_loss_0.004864055672838579_abloss_0.01626750328920594.pth
```
tbdc = 0.465736779771805
rbdc = 0.45988801652655564
Macro AUC: 0.802399797337249
Micro AUC: 0.8323451543068618
```

#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_epoch_1_loss_0.0026884031917704864_abloss_0.003281557291014112
```

Macro AUC:
Micro AUC:
```


#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_epoch_3_loss_0.06606840845113071_abloss_0.11291517147510753.pth
```py
tbdc = 0.47079771157866124
rbdc = 0.5939101875390059
Macro AUC: 0.7980417184575056
Micro AUC: 0.8335901598897388
```

#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_epoch_4_loss_0.04966858399644772_abloss_0.13585608349844752.pth
```py
tbdc = 0.4420123136397074
rbdc = 0.5103994729012586
Macro AUC: 0.8207653342558303
Micro AUC: 0.8268549999718989
```

#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_epoch_5_loss_0.058019471638350854_abloss_0.15550732578499696.pth
```py
tbdc = 0.4432176826282329
rbdc = 0.4665804052330002
Macro AUC: 0.8162550484779343
Micro AUC: 0.8227615974902976
```

#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_epoch_6_loss_0.03635526692497562_abloss_0.09672218865481179
```py
tbdc = 0.42589218877750235
rbdc = 0.5157678458030106
Macro AUC: 0.8076108166633735
Micro AUC: 0.804952751790975
```

#### autoencoder_avenue_ubnormal_adversarial_random_teacher_t12_decoder_yolov8_adamw_epoch_9_loss_0.25919128500701316_abloss_0.9258884270196405.pth
```py
tbdc = 0.4251366084853662
rbdc = 0.34917300530980155
Macro AUC: 0.7419987350095594
Micro AUC: 0.7773412505025763
```




#### 

- With gradient accumulation
```
--- Epoch 1 ---
Epoch 1, Normal Loss: 0.002874622693159732
Epoch 1, Abnormal Loss: 0.01626750328920594
Epoch 1, Overall Loss: 0.004334026287861102
--- Epoch 2 ---
Epoch 2, Normal Loss: 0.0022870854196425996
Epoch 2, Abnormal Loss: 0.003281557291014112
Epoch 2, Overall Loss: 0.0023954516332053617
--- Epoch 3 ---
Epoch 3, Normal Loss: 0.2434410512135432
Epoch 3, Abnormal Loss: 1.5302832827280772
Epoch 3, Overall Loss: 0.3836664617008367
--- Epoch 4 ---
Epoch 4, Normal Loss: 0.05225945676470229
Epoch 4, Abnormal Loss: 0.11291517147510753
Epoch 4, Overall Loss: 0.058869025826186874
--- Epoch 5 ---
Epoch 5, Normal Loss: 0.03305407555114851
Epoch 5, Abnormal Loss: 0.13585608349844752
Epoch 5, Overall Loss: 0.04425626744436843
--- Epoch 6 ---
Epoch 6, Normal Loss: 0.03900171596618771
Epoch 6, Abnormal Loss: 0.15550732578499696
Epoch 6, Overall Loss: 0.051697170468790646
--- Epoch 7 ---
Epoch 7, Normal Loss: 0.024526634706802025
Epoch 7, Abnormal Loss: 0.09672218865481179
Epoch 7, Overall Loss: 0.0323936840268729
--- Epoch 8 ---
Epoch 8, Normal Loss: 0.018355284377868805
Epoch 8, Abnormal Loss: 0.09399173278054951
Epoch 8, Overall Loss: 0.026597283105679156
--- Epoch 9 ---
Epoch 9, Normal Loss: 0.013552534596448005
Epoch 9, Abnormal Loss: 0.09298594199396946
Epoch 9, Overall Loss: 0.022208282592885965
--- Epoch 10 ---
Epoch 10, Normal Loss: 0.1459598357273171
Epoch 10, Abnormal Loss: 0.9258884270196405
Epoch 10, Overall Loss: 0.23094757099055388
--- Epoch 11 ---
Epoch 11, Normal Loss: 0.25889405458940595
Epoch 11, Abnormal Loss: 1.620653931802683
Epoch 11, Overall Loss: 0.40728313642232405
--- Epoch 12 ---
Rest of epochs are the same
Finished Training

```

## Adversarial Training

### Exp 1 -> Close or Worse
- Fixed Gaussian Noise (0, 1.0)
- 0.1 abnormal factor
- normal background
- Adam
- lr 0.001
- combined losses

### Exp 2 -> close or worse
- Fixed Gaussian Noise (0, 4.0)
- 0.1 abnormal factor
- normal background
- Adam
- lr 0.001
- combined losses


### Exp 3 -> close then got worse
- Fixed Gaussian Noise (0, 2.0)
- 0.2 abnormal factor
- normal background
- Adam
- lr 0.001
- combined losses


### Exp 4 -> close, slightly better, then got worse
- Fixed Gaussian Noise (0, 2.0)
- 0.2 abnormal factor
- avenue background
- Adam
- lr 0.001
- combined losses

### Exp 4 -> close or worse
- Fixed Gaussian Noise (0, 1.0)
- 0.2 abnormal factor
- avenue background
- Adam
- lr 0.001
- separate losses

### Exp 5 -> close, slightly better, then got worse
- Fixed Gaussian Noise (0, 1.0)
- 0.02 abnormal factor
- avenue background
- Adam
- lr 0.001
- combined losses

### Exp 6 -> close then got worse
- Fixed Gaussian Noise (0, 1.0)
- 0.02 abnormal factor
- avenue background
- Adam
- lr 0.001
- combined losses
- KL divergence loss as opposed to MSE
