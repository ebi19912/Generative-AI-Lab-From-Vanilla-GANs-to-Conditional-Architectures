# 🧠 Generative AI Lab: From Vanilla GANs to Conditional Architectures

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Open In Colab](https://img.shields.io/badge/Colab-Run%20in%20Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ebi19912/Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures/blob/main/All_gan.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.style=for-the-badge?style=for-the-badge)](LICENSE)

<br/>

**A systematic deep dive into Generative Adversarial Networks (GANs): Progressing from foundational multi-layer perceptron baselines to spatial convolutional stability, semi-supervised classification, and deterministic conditional generation.**

[ English Documentation ](#-table-of-contents) &nbsp;•&nbsp; [ مطالعه مستندات به زبان فارسی ](#-آزمایشگاه-هوش-مصنوعی-مولد-از-gan-ساده-تا-معماریهای-شرطی)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architectures at a Glance](#-architectures-at-a-glance)
- [Detailed Model Implementations](#-detailed-model-implementations)
  - [1. Vanilla GAN (Sequential MLP Baseline)](#1-vanilla-gan-sequential-mlp-baseline)
  - [2. Deep Convolutional GAN (DCGAN)](#2-deep-convolutional-gan-dcgan)
  - [3. Semi-Supervised GAN (SGAN)](#3-semi-supervised-gan-sgan)
  - [4. Conditional GAN (CGAN)](#4-conditional-gan-cgan)
- [Theoretical & Mathematical Foundations](#-theoretical--mathematical-foundations)
- [Training Dynamics & Stabilization Techniques](#-training-dynamics--stabilization-techniques)
- [Benchmark & Experimental Insights](#-benchmark--experimental-insights)
- [Project Structure](#-project-structure)
- [Quick Start & Google Colab Execution](#-quick-start--google-colab-execution)
- [Requirements & Dependencies](#-requirements--dependencies)
- [🇮🇷 بخش فارسی (Persian Documentation)](#-آزمایشگاه-هوش-مصنوعی-مولد-از-gan-ساده-تا-معماریهای-شرطی)
  - [معرفی پروژه](#معرفی-پروژه)
  - [معماری‌های پیاده‌سازی شده](#معماریهای-پیادهسازی-شده)
  - [تئوری و ریاضیات GAN](#تئوری-و-ریاضیات-شبکههای-زاینده-خصمانه)
  - [نتایج و دستاوردهای کلیدی](#نتایج-و-دستاوردهای-کلیدی)
  - [راهنمای اجرا در گوگل کولب و لوکال](#راهنمای-اجرا)
- [License & Citation](#-license--citation)

---

## 🌟 Overview

Generative Adversarial Networks (GANs), originally introduced by Ian Goodfellow et al. (2014), revolutionized generative modeling by framing data generation as a zero-sum game between two competing neural networks:
1. **The Generator ($G$)**: Captures the target data distribution and maps low-dimensional noise vectors to high-dimensional synthetic samples.
2. **The Discriminator ($D$)**: Evaluates incoming samples and estimates the probability that a sample came from real training data rather than the Generator.

This laboratory repository documents the end-to-end implementation, architectural engineering, training dynamics, and empirical evaluation of **four milestone GAN architectures** on the **MNIST benchmark dataset** ($28 \times 28$ grayscale digits).

```
 +------------------+          +---------------+
 |  Gaussian Noise  | -------> | Generator (G) | ------+
 |    z ~ N(0, I)   |          +---------------+       |
 +------------------+                                  v
                                                [ Fake Sample ]
                                                       |
                                                       v
 +------------------+          +-------------------+   |   +---------------------+
 |  Real Dataset    | -------> | Discriminator (D) | <-+-> | Real / Fake Output  |
 |    x ~ P_data    |          +-------------------+       | (or Class Label y)  |
 +------------------+                                      +---------------------+
```

---

## 📊 Architectures at a Glance

| Model | Generator Backbone | Discriminator Backbone | Latent Dim ($z$) | Conditioning ($y$) | Unique Mechanism / Objective |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Vanilla GAN** | Fully Connected (Dense) | Fully Connected (Dense) | $100$ | ❌ None | Baseline minimax game; LeakyReLU + Tanh output |
| **DCGAN** | Conv2DTranspose + BatchNorm | Strided Conv2D + BatchNorm | $100$ | ❌ None | Spatial up/downsampling without pooling; LeakyReLU ($\alpha=0.01$) |
| **SGAN** | Conv2DTranspose + BatchNorm | Strided Conv2D + Dual Head | $100$ | ❌ None | Dual-purpose Discriminator ($K=10$ classes + Unsupervised Real/Fake) |
| **CGAN** | Conv2DTranspose + Embedding | Strided Conv2D + Embedding | $100$ | ✅ Label ($0-9$) | Deterministic class-guided synthesis via embedding projection & channel concat |

---

## 🔬 Detailed Model Implementations

### 1. Vanilla GAN (Sequential MLP Baseline)
The foundational GAN architecture implemented via Keras Sequential API to study the baseline dynamics of non-convex adversarial optimization.

- **Generator ($G$)**:
  - `Dense(128) -> LeakyReLU(alpha=0.01) -> Dense(784, activation='tanh') -> Reshape((28, 28, 1))`
- **Discriminator ($D$)**:
  - `Flatten() -> Dense(128) -> LeakyReLU(alpha=0.01) -> Dense(1, activation='sigmoid')`
- **Key Takeaway**: While MLPs can generate coarse digit patterns, they lack translation invariance and struggle with spatial structural coherence, motivating convolutional generative modeling.

---

### 2. Deep Convolutional GAN (DCGAN)
DCGAN incorporates spatial convolutions and batch normalization principles (Radford et al., 2015) to achieve stable adversarial training and crisp image synthesis.

```
Generator:
  z (100) -> Dense(7*7*256) -> Reshape(7, 7, 256)
          -> Conv2DTranspose(128, k=3, s=2, same) + BN + LeakyReLU(0.01)  --> (14, 14, 128)
          -> Conv2DTranspose(64,  k=3, s=1, same) + BN + LeakyReLU(0.01)  --> (14, 14, 64)
          -> Conv2DTranspose(1,   k=3, s=2, same) + Tanh                   --> (28, 28, 1)

Discriminator:
  x (28, 28, 1) -> Conv2D(32,  k=3, s=2, same) + LeakyReLU(0.01)         --> (14, 14, 32)
                -> Conv2D(64,  k=3, s=2, same) + BN + LeakyReLU(0.01)     --> (7, 7, 64)
                -> Conv2D(128, k=3, s=2, same) + BN + LeakyReLU(0.01)     --> (3, 3, 128)
                -> Flatten() -> Dense(1, activation='sigmoid')
```

- **Key Highlights**:
  - **No Pooling Layers**: Replaced max-pooling with strided convolutions (Discriminator) and fractional-strided convolutions (Generator).
  - **Batch Normalization**: Applied to both networks to stabilize gradient flow and prevent mode collapse.
  - **Output Activation**: Hyperbolic tangent (`tanh`) bounded in $[-1, 1]$ paired with scaled normalized inputs: $x_{norm} = \frac{x - 127.5}{127.5}$.

---

### 3. Semi-Supervised GAN (SGAN)
SGAN turns the Discriminator into a powerful semi-supervised classifier by training on a small subset of labeled data (e.g., **only 100 labeled samples**) alongside abundant unlabeled and generated fake data.

```
                              +-------------------------+
                              | Shared Conv2D Backbone  |
                              | (32 -> 64 -> 128 + Drop)|
                              +-------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
      [ Supervised Head (Softmax) ]                [ Unsupervised Head (Lambda) ]
        Loss: Categorical Cross-Entropy              Loss: Binary Cross-Entropy
        Output: P(y = k | x), k in {0..9}            Output: D(x) = 1 - 1 / (sum(exp(x)) + 1)
```

- **Unsupervised Loss Formulation**:
  The unsupervised probability that a sample is real is computed directly from the unnormalized logits $l(x)$ without adding an explicit $(K+1)$-th class:
  $$D_{\text{unsup}}(x) = 1 - \frac{1}{\sum_{k=1}^K e^{l_k(x)} + 1}$$
- **Empirical Breakthrough**:
  - **Standard Supervised CNN** with 100 samples $\rightarrow$ **~24.65% Test Accuracy** (severe overfitting).
  - **SGAN-Regularized Classifier** with same 100 samples $\rightarrow$ **Substantially Superior Generalization**, proving that adversarial generative modeling serves as an exceptional unsupervised feature prior.

---

### 4. Conditional GAN (CGAN)
Conditional GAN (Mirza & Osindero, 2014) extends GANs by providing external conditioning signals (class labels $y \in \{0, \dots, 9\}$) to both the Generator and Discriminator, turning random sampling into **deterministic, targeted generation**.

```
[ Generator Conditioning ]:
  z (100)       ----+
                    |---> Multiply() ---> Base DCGAN Generator ---> Synthetic Image (28, 28, 1)
  Label y (1)  --> Embedding(10, 100) -+

[ Discriminator Conditioning ]:
  Image (28, 28, 1)   ----+
                          |---> Concatenate(axis=-1) ---> Conv2D Classifier ---> Real / Fake
  Label y (1)        --> Embedding(10, 784) -> Reshape(28, 28, 1) -+
```

- **Key Highlights**:
  - **Generator**: Uses an `Embedding(10, 100)` layer to project discrete labels into latent space, multiplied elementwise (`Multiply`) with noise vector $z$.
  - **Discriminator**: Uses an `Embedding(10, 784)` layer, reshapes it to $(28, 28, 1)$, and concatenates it as an extra channel dimension (`Concatenate(axis=-1)`), resulting in input tensor shape $(28, 28, 2)$.
  - **Result**: Ability to specify any digit ($0$ through $9$) and synthesize high-fidelity variations of that exact digit on command.

---

## 📐 Theoretical & Mathematical Foundations

### The Minimax Objective (Vanilla & DCGAN)
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- **Discriminator Gradient Step**:
  $$\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m \left[ \log D(x^{(i)}) + \log(1 - D(G(z^{(i)}))) \right]$$
- **Generator Gradient Step (Non-saturating heuristic)**:
  $$\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log D(G(z^{(i)}))$$

### The Conditional Minimax Objective (CGAN)
$$\min_G \max_D V(D, G) = \mathbb{E}_{x, y \sim p_{\text{data}}(x, y)}[\log D(x, y)] + \mathbb{E}_{z \sim p_z(z), y \sim p_y(y)}[\log(1 - D(G(z, y), y))]$$

---

## ⚙️ Training Dynamics & Stabilization Techniques

To overcome common GAN pathologies such as **mode collapse**, **vanishing gradients**, and **discriminator overpowering**, the following engineering practices are applied:

1. **Activation Design**:
   - `LeakyReLU(alpha=0.01)` in intermediate layers to maintain non-zero gradient flow for negative activations.
   - `tanh` activation in Generator final layer to map directly to normalized image domain $[-1, 1]$.
   - `sigmoid` activation in Discriminator for probabilistic real/fake scoring.
2. **Batch Normalization**:
   - Applied after Conv2DTranspose and Conv2D layers to normalize internal covariate shift and stabilize layer gradients.
3. **Alternating Batch Optimization**:
   - Discriminator weights are trained on batches of real and generated images (`train_on_batch`).
   - Discriminator is frozen (`trainable = False`) during the combined GAN update step so only Generator weights receive backpropagated gradients.
4. **Data Normalization**:
   - Scaled from integer $[0, 255]$ to floating-point range $[-1.0, +1.0]$ via $X_{\text{norm}} = \frac{X}{127.5} - 1.0$.

---

## 📁 Project Structure

```bash
Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures/
│
├── All_gan.ipynb          # Comprehensive notebook containing all 4 GAN implementations & training loops
├── requirements.txt       # Python package dependencies (TensorFlow, Keras, NumPy, Matplotlib)
├── .gitignore             # Standard gitignore for Python, Jupyter checkpoints, and environments
└── README.md              # Full project documentation in English and Persian
```

---

## 🚀 Quick Start & Google Colab Execution

### Option 1: Run in Google Colab (Recommended)
Launch the interactive notebook directly in Google Colab with GPU acceleration:

[![Open In Colab](https://img.shields.io/badge/Colab-Run%20in%20Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ebi19912/Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures/blob/main/All_gan.ipynb)

*Tip: In Google Colab, go to `Runtime` > `Change runtime type` > Select `T4 GPU` for accelerated training.*

### Option 2: Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ebi19912/Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures.git
   cd Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Lab / Notebook**:
   ```bash
   jupyter notebook All_gan.ipynb
   ```

---

## 📦 Requirements & Dependencies

- **Python**: $\ge 3.8$
- **TensorFlow**: $\ge 2.10.0$
- **NumPy**: $\ge 1.23.0$
- **Matplotlib**: $\ge 3.6.0$
- **Jupyter / IPykernel**: Latest

---

<br/>

---

<div dir="rtl">

# 🇮🇷 آزمایشگاه هوش مصنوعی مولد: از GAN ساده تا معماری‌های شرطی

## معرفی پروژه
این مخزن شامل پیاده‌سازی گام‌به‌گام و تحلیل تجربی **۴ معماری جریان‌ساز از شبکه‌های زایای خصمانه (GANs)** بر روی مجموعه داده استاندارد **MNIST** با فریم‌ورک **TensorFlow / Keras** می‌باشد. هدف این پروژه، بررسی مسیر تکامل شبکه‌های مولد از مدل‌های ابتدایی مبتنی بر پرسپترون چندلایه (MLP) تا ساختارهای پیشرفته کانوولوشنی، یادگیری نیمه‌نظارتی و تولید کنترل‌شده‌ی تصاویر شرطی است.

---

## معماری‌های پیاده‌سازی شده

### ۱. شبکه زایای ساده (Vanilla GAN)
- **ساختار ژنراتور**: لایه‌های متراکم (`Dense`) به همراه تابع فعالیت `LeakyReLU` و لایه نهایی `tanh` برای تولید بردار تصویر $28 \times 28$.
- **ساختار دیسکریمناتور**: لایه `Flatten` و لایه‌های `Dense` با فعال‌ساز `sigmoid` برای تخمین احتمال اصالت داده (واقعی یا ساختگی).
- **هدف**: آشنایی با بازی مینی‌ماکس پایه و چالش‌های بهینه‌سازی غیرمحدب در فضاهای خطی.

### ۲. شبکه زایای کانوولوشنی عمیق (DCGAN)
- **ساختار ژنراتور**: استفاده از لایه‌های کانوولوشن ترانهاده (`Conv2DTranspose`) جهت افزایش مقیاس فضایی تصویر از $7 \times 7$ به $14 \times 14$ و در نهایت $28 \times 28$ همراه با `BatchNormalization`.
- **ساختار دیسکریمناتور**: کانوولوشن‌های دارای گام (`strided Conv2D`) به جای لایه‌های Max Pooling به همراه لایه‌های نرمال‌سازی دسته‌ای.
- **هدف**: حل معضل محوشدگی گرادیان و دستیابی به کیفیت بصری بسیار بالاتر و حفظ پیوستگی فضایی پیکسل‌ها.

### ۳. شبکه زایای نیمه‌نظارتی (Semi-Supervised GAN - SGAN)
- **ساختار**: بهره‌گیری از یک دیسکریمناتور مشترک با دو سر خروجی:
  ۱. **سر نظارتی (Supervised Head)**: دسته‌بندی ۱۰ کلاسه ارقام با تابع `Softmax`.
  ۲. **سر غیرنظارتی (Unsupervised Head)**: تشخیص داده واقعی/جعلی با تابع اختصاصی `Lambda`.
- **دستاورد کلیدی**: نشان داده شد که با استفاده از تنها **۱۰۰ نمونه برچسب‌دار** (و استفاده از سایر داده‌ها به صورت بدون برچسب در فرآیند خصمانه)، عملکرد دسته‌بند به مراتب بهتر از یک مدل عادی عمل می‌کند (دقت مدل عادی تنها حدود ۲۴.۶٪ بود).

### ۴. شبکه زایای شرطی (Conditional GAN - CGAN)
- **ساختار**: تزریق بردار برچسب هدف (کلاس $0$ تا $9$) به هر دو شبکه ژنراتور و دیسکریمناتور:
  - در **ژنراتور**: نگاشت برچسب به فضای ویژگی با `Embedding` و ضرب برداری در نویز اولیه $z$.
  - در **دیسکریمناتور**: نگاشت برچسب به ابعاد تصویر با لایه امبدینگ و الصاق آن به عنوان کانال دوم (`Concatenate` روی بعد کانال).
- **هدف**: امکان تولید دقیق و قطعی یک رقم مشخص بر اساس خواسته کاربر به جای تولید ارقام تصادفی.

---

## تئوری و ریاضیات شبکه‌های زاینده خصمانه

تابع هدف بازی دونفره مینی‌ماکس در GAN:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

در شبکه شرطی (CGAN)، توزیع شرطی روی برچسب $y$ اعمال می‌گردد:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{z, y}[\log(1 - D(G(z, y), y))]$$

---

## نتایج و دستاوردهای کلیدی
- **تثبیت فرآیند آموزش**: استفاده از بهینه‌ساز Adam با نرخ یادگیری متناسب و توابع فعال‌ساز LeakyReLU ($\\alpha=0.01$).
- **حل مشکل Mode Collapse**: استفاده از Batch Normalization در معماری DCGAN.
- **تولید هدایت‌شده**: تولید رقمی خاص با کیفیت بالا به کمک امبدینگ‌های شرطی در CGAN.
- **بهره‌وری داده‌های بدون برچسب**: اثبات قدرت مدل‌سازی مولد در یادگیری نیمه‌نظارتی (SGAN).

---

## راهنمای اجرا

### اجرا در محیط گوگل کولب (سریع و با پردازنده گرافیکی GPU)
کافیست بر روی نشان زیر کلیک نمایید تا مستقیماً نوت‌بوک پروژه در Google Colab باز شود:

[![Open In Colab](https://img.shields.io/badge/Colab-Run%20in%20Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ebi19912/Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures/blob/main/All_gan.ipynb)

### اجرای محلی
```bash
git clone https://github.com/ebi19912/Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures.git
cd Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures
pip install -r requirements.txt
jupyter notebook All_gan.ipynb
```

</div>

---

## 📜 License & Citation

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute this codebase for educational and research purposes.

```bibtex
@misc{ebrahimi2026generativeailab,
  author = {Ebrahimi},
  title = {Generative AI Lab: From Vanilla GANs to Conditional Architectures},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ebi19912/Generative-AI-Lab-From-Vanilla-GANs-to-Conditional-Architectures}}
}
```

<div align="center">
⭐ If you find this laboratory repository helpful, please consider giving it a star! ⭐
</div>
