# Additions to the Updated Proposal doc
1. Updated the Introduction Section
2. Added Literature Review Section with updated references


## Existing Libraries

https://pypi.org/project/nsfw-detector/

# Reading Material:
### Policy:
https://transparency.fb.com/en-gb/policies/community-standards/adult-nudity-sexual-activity/
https://transparency.fb.com/policies/improving/proactive-rate-metric/

### Research Paper for Meta's NSFW detection model:

https://about.fb.com/news/2021/12/metas-new-ai-system-tackles-harmful-content/
https://ai.meta.com/blog/harmful-content-can-evolve-quickly-our-new-ai-system-adapts-to-tackle-it/

### Bumble Open Source NSFW Detector

https://bumble.com/en-us/the-buzz/bumble-open-source-private-detector-ai-cyberflashing-dick-pics  (Article)  
https://github.com/bumble-tech/private-detector (Repo)


### Generate Dataset (using GenAI)
https://arxiv.org/pdf/2210.08402.pdf (Prof. recommended)  
https://rom1504.github.io/clip-retrieval/ (Clip embeddings, search interface)

### Others
https://github.com/EBazarov/nsfw_data_source_urls (Dataset)  
https://github.com/yangbisheng2009/nsfw-resnet (PyTorch Resnet)  

## Black-box label-only attacks (only from decent conferences)  
Kamalnath -   
Sign-OPT: A Query-Efficient Hard-label Adversarial Attack - https://arxiv.org/abs/1909.10773  (ICLR '20)  
Query-Efficient Hard-label Black-box Attack - https://arxiv.org/pdf/1807.04457.pdf  (ICLR '19)  
Improving Query Efficiency of Black-box Adversarial Attack - https://arxiv.org/pdf/2009.11508.pdf (ECCV '19)  
Blackbox Attacks via Surrogate Ensemble Search - https://arxiv.org/pdf/2208.03610.pdf (NIPS '22)  

Jagadeesh -   
Improving Black-box Adversarial Attacks with a Transfer-based Prior - https://arxiv.org/abs/1906.06919 (NIPS '19)  
Low-Frequency Adversarial Perturbation - https://arxiv.org/abs/1809.08758 (PMLR '19)  
Square Attack: a query-efficient black-box adversarial attack via random search - https://arxiv.org/abs/1912.00049 (ECCV '20)  
Structure Invariant Transformation for better Adversarial Transferability - https://arxiv.org/pdf/2309.14700.pdf (ICCV '23)  

Neelesh -  
Improving Adversarial Transferability via Neuron Attribution-Based Attacks - https://arxiv.org/abs/2204.00008  (CVPR '22)  
Patch-wise Attack for Fooling Deep Neural Network - https://arxiv.org/pdf/2007.06765.pdf (ECCV '20)  
Triangle Attack: A Query-efficient Decision-based Adversarial Attack - https://arxiv.org/abs/2112.06569 (ECCV '22)  
Towards Efficient Data Free Black-box Adversarial Attack - https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Towards_Efficient_Data_Free_Black-Box_Adversarial_Attack_CVPR_2022_paper.pdf (CVPR '22)  

Two types of black-box attacks - Decision-based and Transfer-based  

### Decision-based  
[Sign-OPT: A Query-Efficient Hard-label Adversarial Attack](https://arxiv.org/abs/1909.10773) (ICLR '20)  
[Query-Efficient Hard-label Black-box Attack](https://arxiv.org/pdf/1807.04457.pdf) (ICLR '19)  
[Improving Query Efficiency of Black-box Adversarial Attack](https://arxiv.org/pdf/2009.11508.pdf) (ECCV '19)   
[Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) (ECCV '20)  
[Triangle Attack: A Query-efficient Decision-based Adversarial Attack](https://arxiv.org/abs/2112.06569) (ECCV '22)  
[Low-Frequency Adversarial Perturbation](https://arxiv.org/abs/1809.08758) (PMLR '19)  
[Blackbox Attacks via Surrogate Ensemble Search](https://proceedings.neurips.cc/paper_files/paper/2022/file/23b9d4e18b151ba2108fb3f1efaf8de4-Paper-Conference.pdf) (NIPS '22)

### Transfer-based
[Improving Adversarial Transferability via Neuron Attribution-Based Attacks](https://arxiv.org/abs/2204.00008)  (CVPR '22)  
[Improving Black-box Adversarial Attacks with a Transfer-based Prior](https://arxiv.org/abs/1906.06919) (NIPS '19)  
[Patch-wise Attack for Fooling Deep Neural Network](https://arxiv.org/pdf/2007.06765.pdf) (ECCV '20)  
[Towards Efficient Data Free Black-box Adversarial Attack](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Towards_Efficient_Data_Free_Black-Box_Adversarial_Attack_CVPR_2022_paper.pdf) (CVPR '22)  
[Structure Invariant Transformation for better Adversarial Transferability](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Structure_Invariant_Transformation_for_better_Adversarial_Transferability_ICCV_2023_paper.pdf) (ICCV '23)

