# Code Repository for Knowledge Enhanced Machine Learning Pipeline against Diverse Adversarial Attacks

* Complete repository with full data and pretrained models : 

  https://www.dropbox.com/sh/4xxz10xxi0o5kjb/AAAhkYrCtHWD-huP3D_FCK1Ga?dl=0

* Evaluation against $\mathcal{L}_\infty$, unforeseen attacks and common corruptions : `./SettingA-linf,unforeseen,corruptions`

  * Setup 

    * Download `./SettingA-linf,unforeseen,corruptions/data/data` from the dropbox directory, which contains all the clean data, pregenerated adversarial examples, preporcessed sensory information.

    * Download `./SettingA-linf,unforeseen,corruptions/pipeline/sensor` from the dropbox directory, which contains all the sensors (submodels) for KEMLP (SettingA).

  * Reimplement tables : Please refer to `./SettingA-linf,unforeseen,corruptions/readme.md`

* Evaluation against physical stop sign attack : `./SettingB-stop_sign_attack`

  * Setup 
    * Download `./SettingB-stop_sign_attack/data/data` from the dropbox directory, which contains all the clean data, pregenerated adversarial examples, preporcessed sensory information.
    * Download `./SettingB-stop_sign_attack/pipeline/sensor` from the dropbox directory, which contains all the sensors (submodels) for KEMLP (SettingB).
  * Reimplement tables : Please refer to `./SettingB-stop_sign_attack/readme.md`
  
* Environment Dependency

  * python 3.6
  * pytorch 1.7.0
  * cv2