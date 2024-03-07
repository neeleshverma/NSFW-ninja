# ByPassing NSFW Gatekeepers (Black-box setting)

This project explores and exploits the content moderation filters on social media platforms. The study delves deep into the vulnerabilities of these filters, employing a systematic “black-box” attack approach. We unveil an innovative technique harnessing Grad-CAM heatmaps, which highlight key pixels crucial for image classification as NSFW or not. Armed with this knowledge, we strategically inject calculated noise into these areas, crafting “adversarial attacks” that bypass the filters’ defenses. By systematically testing the attack on popular social media platforms - Bumble and Reddit, we expose the potential for misuse and raise critical questions about the effectiveness and reliability of content moderation systems.

Final Report - [here](https://github.com/neeleshverma/NSFW-ninja/blob/master/final-report/CSE509_FinalReport.pdf)  
Code is currently being updated.   

### 1. Training Helper Models
Currently, we are using 3 types of models - Resnets (34, 50), Inception-v3, and ViT. To train on a new dataset, simply modify ``configs/resnet_configs.yaml`` or make a new yaml file (need to change it in the code as well).
```
$ cd code
$ python resnet_train.py
```
Feel free to change/tune the hyper-parameters in the config file.  

### 2. Running Grad-CAM attack
```
$ python grad_cam_attack.py
```
We are using Gaussian noises, so the parameters of the Gaussian can be fine-tuned in this Python file.

### 3. Comparison with PGD, Auto-PGD, and C&W attack
We have also implemented PGD, Auto-PGD, and C&W attacks using the [ART](https://adversarial-robustness-toolbox.readthedocs.io/) library. They can be run as (for resnet) -
```
$ python PGD_attack/pgd_resnet.py
$ python PGD_attack/autopgd.py
$ python CandW/cw_resnet.py
```

We will update the comparison tables here.
