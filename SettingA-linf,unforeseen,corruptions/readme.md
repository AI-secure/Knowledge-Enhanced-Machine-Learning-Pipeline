

# Setting-A

This directory contains the code for evaluating our KEMLP framework against :

* **$L_\infty$ attack**

  <img src="img\0.png" width="45%" />

* **Unforeseen attacks**

  <img src="img\14.png" width="100%" />

* **Natural corruption**

  <img src="img\15.png" width="100%" />

<br><br>

## Reimplement $L_\infty$ Attack

For $L_{\infty}$ attack, we generate the adversarial examples under three different attack settings:

* **Whitebox sensor attack**

  <img src="img\16.png" width="100%" />

  ```bash
  cd pipeline
  python reimplement_table_linf_whitebox.py
  ```

* **Blackbox sensor attack**

  <img src="img\8.png" alt="image-20210211194144345"  />

  ```bash
  cd pipeline
  python reimplement_table_linf_blackbox_sensor.py
  ```

* **Blackbox pipeline attack**

  <img src="img\9.png" alt="image-20210211194220988"  />
  
    ```bash
    cd pipeline
    python reimplement_table_linf_blackbox_pipeline.py
    ```

<br><br>

## Reimplement Unforeseen Attacks

* **Whitebox sensor attack**

  <img src="img\17.png" alt="image-20210211194220988"  />

  ```bash
    cd pipeline
    python reimplement_table_unforeseen_whitebox.py
  ```

* **Blackbox sensor attack**

  <img src="img\11.png" alt="image-20210211205621986"  />

  ```bash
    cd pipeline
    python reimplement_table_unforeseen_blackbox_sensor.py
  ```

* **Blackbox pipeline attack**

  <img src="img\12.png" alt="image-20210211205742555"  />

  ```bash
    cd pipeline
    python reimplement_table_unforeseen_blackbox_pipeline.py
  ```

<br><br>

## Reimplement Common Corruptions

<img src="img\13.png" alt="image-20210211210026437"  />

```bash
  cd pipeline
  python reimplement_table_common_corruption.py
```

