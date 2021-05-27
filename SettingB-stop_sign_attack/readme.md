# Setting-B : Stop Sign Attack

This directory contains the code for evaluating our KEMLP framework against **physically implementable stop sign attack**. ([[1]](#ref-1),[[2]](#ref-2))

<img src="img\0.png" width="45%" />

(**left:** the attacked stop sign sample; **middle:** the extracted border information from the attacked sample; **right:** the extracted content information from the attacked sample)

To be consistent with [[1]](#ref-1), we use the same dataset (a modified version of GTSRB dataset where the German stop signs are replaced with US stop signs from LISA dataset), GTSRB-CNN model (the same architecture and the same parameters), stop sign samples, and adversarial stickers from [[1]](#ref-1). 

<br><br>

## Reimplement the table

![image-20210211193002455](img\1.png)

```bash
cd pipeline
python reimplement_table.py
```

<br><br>

## Reference

<span id='ref-1'>[1] Eykholt, Kevin, et al. "Robust physical-world attacks on deep learning visual classification." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018.</span>

<span id='ref-2'>[2] Wu, Tong, Liang Tong, and Yevgeniy Vorobeychik. "Defending against physically realizable attacks on image classification." *arXiv preprint arXiv:1909.09552* (2019).</span>

