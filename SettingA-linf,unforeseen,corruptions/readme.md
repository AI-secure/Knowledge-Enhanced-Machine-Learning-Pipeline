

# Setting-A

This directory contains the code for evaluating our KEMLP framework against :

* **$L_\infty$ attack**

  <img src="img\0.png" width="45%" />

* **Unforeseen attacks**

  <img src="img\1.png" width="33%" /><img src="img\2.png" width="33%" /><img src="img\3.png" width="33%" />

* **Natural corruption**

  <img src="img\4.png" width="33%" /><img src="img\5.png" width="33%" /><img src="img\6.png" width="33%" />

<br><br>

## Reimplement $L_\infty$ Attack

For $L_{\infty}$ attack, we generate the adversarial examples under three different attack settings:

* **Whitebox sensor attack**

  ![image-20210211193830941](img\7.png)

  ```bash
  cd pipeline
  python reimplement_table_linf_whitebox.py
  ```

* **Blackbox sensor attack**

  ![image-20210211194144345](img\8.png)

  ```bash
  cd pipeline
  python reimplement_table_linf_blackbox_sensor.py
  ```

* **Blackbox pipeline attack**

  ![image-20210211194220988](img\9.png)
  
    ```bash
    cd pipeline
    python reimplement_table_linf_blackbox_pipeline.py
    ```

<br><br>

## Reimplement Unforeseen Attacks

* **Whitebox sensor attack**

  ![image-20210211204347658](img\10.png)

  ```bash
    cd pipeline
    python reimplement_table_unforeseen_whitebox.py
  ```

* **Blackbox sensor attack**

  ![image-20210211205621986](img\11.png)

  ```bash
    cd pipeline
    python reimplement_table_unforeseen_blackbox_sensor.py
  ```

* **Blackbox pipeline attack**

  ![image-20210211205742555](img\12.png)

  ```bash
    cd pipeline
    python reimplement_table_unforeseen_blackbox_pipeline.py
  ```

<br><br>

## Reimplement Common Corruptions

![image-20210211210026437](img\13.png)

```bash
  cd pipeline
  python reimplement_table_common_corruption.py
```

