# Numerical_Computing_Calculator

# 🧮 NumComp Pro 

**NumComp Pro** is a full-featured numerical computation web app built with **Streamlit**, providing safe AST-based evaluation of mathematical expressions, root-finding method comparisons, and Lagrange interpolation. It’s designed for students, engineers, and anyone interested in numerical methods.

---

## 🚀 Features

### 1. Root-Finding (Comparison)

* Solve equations **f(x) = 0** using multiple methods:

  * **Bisection**
  * **False Position (Regula Falsi)**
  * **Newton-Raphson**
  * **Secant**
  * **Fixed Point**
* Auto-bracketing support to find intervals where roots exist.
* Convergence and function plots with iteration markers.
* Logs for each method with CSV export.
* Safe **AST-based evaluator** (no unsafe `eval`).

### 2. Lagrange Interpolation

* Interpolate data points manually or via CSV upload.
* Visualize interpolation curve and data points.
* CSV export of points and results.

---

## ⚙️ Installation & Usage

1. **Clone the repository**

```bash
git clone https://github.com/AbdulWasayTabba/your-repo-name.git
cd your-repo-name
```

2. **Install dependencies**

```bash
pip install streamlit pandas numpy plotly
```

3. **Run the app**

```bash
streamlit run numcomp_pro_streamlit.py
```

4. Open the provided local URL in your browser (usually `http://localhost:8501`).

---

## 📝 Usage Tips

* Use **math functions** safely via `math.` or `np.`, e.g., `sin(x)` → `math.sin(x)` or `np.sin(x)`.
* Functions can use `^` for exponentiation (automatically converted to `**`).
* For root-finding, provide either:

  * Interval `[a, b]` for Bisection/False Position.
  * Initial guesses for Newton/Secant/Fixed Point.
* Lagrange interpolation can take manual input or CSV with `x, y` columns.

---

## 📊 Visualization

* Function plots with iteration markers.
* Convergence plots: |residual| vs iteration.
* Lagrange interpolation curves with original data points.

---

## 🛡️ Security

* Uses **AST-based expression evaluation** to prevent unsafe code execution.
* No direct `eval` or execution of arbitrary Python code.

---

## 🔗 Example Functions

```
x**3 - x - 2
sin(x) - 0.5
exp(x) - 2
```

* Optional Fixed Point g(x): `(x + 2)**(1/3)`

---

## 💾 Export & Logs

* Download iteration logs as CSV per method.
* Export Lagrange points to CSV.

---

## 🖥️ Screenshots

*<img width="1891" height="902" alt="image" src="https://github.com/user-attachments/assets/0804fe1d-2655-4401-ac03-f0322cb62198" />
*
*<img width="1919" height="712" alt="image" src="https://github.com/user-attachments/assets/d5a4ea68-a28b-46a3-9dfc-bac5485e51c5" />
*
*<img width="1590" height="458" alt="image" src="https://github.com/user-attachments/assets/48739a66-0144-410e-9c3d-bcbcb7f6040b" />
*
*<img width="1560" height="673" alt="image" src="https://github.com/user-attachments/assets/25c9069a-cdcf-4b8b-b639-472cf1256a07" />
*
*<img width="1586" height="698" alt="image" src="https://github.com/user-attachments/assets/0768422f-2e04-46d4-9602-a971239a4a59" />
*
*<img width="1592" height="510" alt="image" src="https://github.com/user-attachments/assets/ede355ff-dbbf-4726-b3c4-07063ae7198f" />
*
---

## 📚 References

* [Numerical Methods for Engineers](https://www.amazon.com/Numerical-Methods-Engineers-Steven-C-Chapra/dp/0073401064)
* Python `ast` documentation for safe evaluation.
* [Streamlit Official Documentation](https://docs.streamlit.io/)

---

## 👤 Author

**Abdulwasay Tabba**

* GitHub: [https://github.com/AbdulWasayTabba](https://github.com/AbdulWasayTabba)

---
