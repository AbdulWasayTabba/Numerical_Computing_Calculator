# numcomp_pro_streamlit.py
# Option B — Full Pro Version (safe AST evaluator, bisection, false position,
# Newton, Secant, Fixed Point, Lagrange, logging, CSV export, convergence plots)
# Run: streamlit run numcomp_pro_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import ast
import operator as op
import base64
import html

# -------------------------
# Safe AST-based evaluator
# -------------------------
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.BitXor: op.pow,  # allow ^ as pow (mapped)
}

_SAFE_NAMES = {
    **{k: getattr(math, k) for k in dir(math) if not k.startswith("__")},
    "np": np,
    "math": math,
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "e": math.e,
    "pi": math.pi,
}


class ExprEvaluator(ast.NodeVisitor):
    def __init__(self, node, names=None):
        self.node = node
        self.names = {} if names is None else names

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        return super().visit(node)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Operator {op_type} not allowed")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](operand)
        raise ValueError("Unary operator not allowed")

    def visit_Num(self, node):
        return node.n

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")

    def visit_Name(self, node):
        if node.id in self.names:
            return self.names[node.id]
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Name '{node.id}' is not allowed")

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        # no keywords allowed
        try:
            return func(*args)
        except Exception as e:
            raise ValueError(f"Error calling function: {e}")

    def generic_visit(self, node):
        raise ValueError(f"Node type {type(node).__name__} not allowed in expression")


def compile_expr_to_func(expr_str):
    expr_str = expr_str.replace("^", "**")
    try:
        parsed = ast.parse(expr_str, mode="eval")
        # reject potentially unsafe nodes
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Assign, ast.Lambda)):
                return None, "Unsafe expression (forbidden node)"
        def f(x):
            evaluator = ExprEvaluator(parsed, names={**_SAFE_NAMES, "x": x})
            return evaluator.visit(parsed)
        # quick sanity test
        _ = f(1.0)
        return f, None
    except Exception as e:
        return None, str(e)

# -------------------------
# Numerical solvers (with logs)
# -------------------------

def derivative_central(func, x, h=1e-6):
    try:
        return (func(x+h) - func(x-h)) / (2*h)
    except Exception:
        return None

def solve_bisection(func, a, b, tol, max_iter, progress=None):
    logs = []
    try:
        fa = func(a)
        fb = func(b)
    except Exception:
        return None, "Eval Error", 0, logs
    if fa is None or fb is None:
        return None, "Eval Error", 0, logs
    if fa * fb > 0:
        return None, "Root not bracketed", 0, logs

    it = 0
    while (b - a)/2 > tol and it < max_iter:
        c = (a + b)/2.0
        try:
            fc = func(c)
        except Exception:
            return None, "Eval Error", it, logs
        logs.append({"iter": it+1, "a": a, "b": b, "c": c, "f(c)": fc, "interval": b-a})
        if abs(fc) < tol:
            return c, "Success", it+1, logs
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        it += 1
        if progress:
            progress(it, max_iter)
    c = (a + b)/2.0
    status = "Success" if (b - a)/2 <= tol else "Max Iter"
    return c, status, it, logs

def solve_false_position(func, a, b, tol, max_iter, progress=None):
    logs = []
    try:
        fa = func(a)
        fb = func(b)
    except Exception:
        return None, "Eval Error", 0, logs
    if fa is None or fb is None:
        return None, "Eval Error", 0, logs
    if fa * fb > 0:
        return None, "Root not bracketed", 0, logs

    it = 0
    c = a
    while it < max_iter:
        try:
            fa = func(a); fb = func(b)
        except Exception:
            return None, "Eval Error", it, logs
        if fb - fa == 0:
            return None, "Div by zero", it, logs
        c = (a*fb - b*fa)/(fb - fa)
        try:
            fc = func(c)
        except Exception:
            return None, "Eval Error", it, logs
        logs.append({"iter": it+1, "a": a, "b": b, "c": c, "f(c)": fc})
        if abs(fc) < tol:
            return c, "Success", it+1, logs
        if fa * fc < 0:
            b = c
        else:
            a = c
        it += 1
        if progress:
            progress(it, max_iter)
    return c, "Max Iter", it, logs

def solve_newton(func, x0, tol, max_iter, progress=None):
    logs = []
    x = x0
    it = 0
    while it < max_iter:
        try:
            fx = func(x)
        except Exception:
            return None, "Eval Error", it, logs
        dfx = derivative_central(func, x)
        logs.append({"iter": it+1, "x": x, "f(x)": fx, "f'(x)": dfx})
        if fx is None or dfx is None:
            return None, "Eval Error", it, logs
        if dfx == 0:
            return None, "Zero derivative", it, logs
        x_new = x - fx/dfx
        if abs(x_new - x) < tol:
            return x_new, "Success", it+1, logs
        x = x_new
        it += 1
        if progress:
            progress(it, max_iter)
    return x, "Max Iter", it, logs

def solve_secant(func, x0, x1, tol, max_iter, progress=None):
    logs = []
    it = 0
    while it < max_iter:
        try:
            fx0 = func(x0); fx1 = func(x1)
        except Exception:
            return None, "Eval Error", it, logs
        logs.append({"iter": it+1, "x0": x0, "x1": x1, "f(x0)": fx0, "f(x1)": fx1})
        if fx0 is None or fx1 is None:
            return None, "Eval Error", it, logs
        denom = fx1 - fx0
        if denom == 0:
            return None, "Div by zero", it, logs
        x2 = x1 - fx1*(x1 - x0)/denom
        if abs(x2 - x1) < tol:
            return x2, "Success", it+1, logs
        x0, x1 = x1, x2
        it += 1
        if progress:
            progress(it, max_iter)
    return x1, "Max Iter", it, logs

def solve_fixed_point(gfunc, x0, tol, max_iter, progress=None):
    logs = []
    x = x0
    it = 0
    while it < max_iter:
        try:
            x_new = gfunc(x)
        except Exception:
            return None, "Eval Error", it, logs
        logs.append({"iter": it+1, "x": x, "g(x)": x_new})
        if x_new is None:
            return None, "Eval Error", it, logs
        if abs(x_new - x) < tol:
            return x_new, "Success", it+1, logs
        if abs(x_new) > 1e12:
            return None, "Diverged", it, logs
        x = x_new
        it += 1
        if progress:
            progress(it, max_iter)
    return x, "Max Iter", it, logs

def lagrange_interpolate(xs, ys, xv):
    n = len(xs)
    result = 0.0
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i != j:
                denom = xs[i] - xs[j]
                if denom == 0:
                    return None
                term *= (xv - xs[j]) / denom
        result += term
    return result

# -------------------------
# Helpers: bracketing, download
# -------------------------
def try_auto_bracket(func, start, end, splits=100):
    xs = np.linspace(start, end, splits)
    try:
        vals = [func(x) for x in xs]
    except Exception:
        return None
    for i in range(len(xs)-1):
        if vals[i] is None or vals[i+1] is None:
            continue
        if vals[i]*vals[i+1] <= 0:
            return float(xs[i]), float(xs[i+1])
    return None

def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

def download_link_bytes(content_bytes: bytes, filename: str, label: str):
    b64 = base64.b64encode(content_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="NumComp Pro (Option B)", page_icon="🧮", layout="wide")

st.markdown("""
<style>
.card { background:#ffffff; padding:14px; border-radius:8px; box-shadow: 0 6px 18px rgba(0,0,0,0.06);}
.func-box { font-family: monospace; background:#f7f7f9; padding:8px; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

st.title("🧮 NumComp Pro — (Option B) Safer · Pro UI · Logs")
st.markdown("Compare root-finding methods and do Lagrange interpolation. AST-based evaluator (safer than raw eval).")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Root Finding (Comparison)", "Lagrange Interpolation"])
    st.markdown("---")
    tol = st.number_input("Tolerance (ε)", value=1e-6, format="%.8f")
    max_iter = st.number_input("Max Iterations", value=200, min_value=1, max_value=5000)
    show_logs = st.checkbox("Show logs", value=True)
    st.markdown("---")
    st.markdown("Examples:")
    st.code("x**3 - x - 2")
    st.code("sin(x) - 0.5")
    st.caption("Use math. or np. for advanced functions, e.g. sin(x) or exp(x).")

# Session state
if "last_results" not in st.session_state:
    st.session_state["last_results"] = {}

# -------------------------
# Root Finding Mode
# -------------------------
if mode == "Root Finding (Comparison)":
    st.header("🔍 Root Finding — Compare Methods")

    c1, c2 = st.columns([2, 1])
    with c1:
        f_str = st.text_area("Enter f(x) = 0", value="x**3 - x - 2", height=100)
        g_str = st.text_input("Optional g(x) for Fixed Point", value="(x + 2)**(1/3)")
        st.markdown("Function preview:")
        st.markdown(f'<div class="func-box">{html.escape(f_str)}</div>', unsafe_allow_html=True)

        st.markdown("Select methods to run:")
        run_bis = st.checkbox("Bisection", value=True)
        run_fp = st.checkbox("False Position (Regula Falsi)", value=True)
        run_new = st.checkbox("Newton-Raphson", value=True)
        run_sec = st.checkbox("Secant", value=True)
        run_fix = st.checkbox("Fixed Point", value=False)

        st.markdown("---")
        st.markdown("Interval / initial guesses")
        a_val = st.number_input("a (or guess 1)", value=1.0)
        b_val = st.number_input("b (or guess 2)", value=2.0)
        newton_guess = st.number_input("Newton initial guess", value=float(a_val))
        sec_x0 = st.number_input("Secant x0", value=float(a_val))
        sec_x1 = st.number_input("Secant x1", value=float(b_val))

        st.markdown("---")
        st.markdown("Auto-bracketing (if a,b don't bracket)")
        bracket_start = st.number_input("Bracket start", value=-10.0)
        bracket_end = st.number_input("Bracket end", value=10.0)
        do_auto = st.checkbox("Try auto-bracket", value=True)

    with c2:
        st.markdown("Run actions")
        run_btn = st.button("🚀 Run selected methods")
        st.markdown("---")
        st.markdown("Export: logs download after run (CSV).")

    # compile
    f_func, f_err = compile_expr_to_func(f_str)
    g_func, g_err = (None, None)
    if g_str.strip():
        g_func, g_err = compile_expr_to_func(g_str)

    if f_err:
        st.error(f"f(x) compile error: {f_err}")
    if g_str.strip() and g_err:
        st.error(f"g(x) compile error: {g_err}")

    if run_btn and not f_err:
        # prepare progress bar
        prog_placeholder = st.empty()
        prog = prog_placeholder.progress(0)
        step_counter = {"count": 0}
        total_est = max_iter * (1 + run_bis + run_fp + run_new + run_sec + run_fix)

        def progress_callback(it, maxit):
            # safe update using mutable dict
            step_counter["count"] += 1
            val = int(min(100, (step_counter["count"] / max(1, total_est)) * 100))
            prog.progress(val)

        a = float(a_val); b = float(b_val)
        # check bracketing
        try:
            fa = f_func(a); fb = f_func(b)
        except Exception:
            fa = fb = None

        if do_auto and (fa is None or fb is None or fa * fb > 0):
            bracket = try_auto_bracket(f_func, float(bracket_start), float(bracket_end), splits=200)
            if bracket:
                st.info(f"Auto-bracketing found [{bracket[0]:.6g}, {bracket[1]:.6g}] — will use this for bracketing methods.")
                a, b = bracket
            else:
                st.warning("Auto-bracketing did not find a sign change in the given range.")

        results = []
        logs_store = {}

        if run_bis:
            root, status, iters, logs = solve_bisection(f_func, a, b, tol, int(max_iter), progress=progress_callback)
            results.append({"Method": "Bisection", "Root Found": root, "Iterations": iters, "Status": status})
            logs_store["Bisection"] = logs

        if run_fp:
            root, status, iters, logs = solve_false_position(f_func, a, b, tol, int(max_iter), progress=progress_callback)
            results.append({"Method": "False Position", "Root Found": root, "Iterations": iters, "Status": status})
            logs_store["False Position"] = logs

        if run_new:
            root, status, iters, logs = solve_newton(f_func, float(newton_guess), tol, int(max_iter), progress=progress_callback)
            results.append({"Method": "Newton-Raphson", "Root Found": root, "Iterations": iters, "Status": status})
            logs_store["Newton-Raphson"] = logs

        if run_sec:
            root, status, iters, logs = solve_secant(f_func, float(sec_x0), float(sec_x1), tol, int(max_iter), progress=progress_callback)
            results.append({"Method": "Secant", "Root Found": root, "Iterations": iters, "Status": status})
            logs_store["Secant"] = logs

        if run_fix and g_func:
            root, status, iters, logs = solve_fixed_point(g_func, a, tol, int(max_iter), progress=progress_callback)
            results.append({"Method": "Fixed Point", "Root Found": root, "Iterations": iters, "Status": status})
            logs_store["Fixed Point"] = logs

        prog.progress(100)
        prog_placeholder.empty()

        st.session_state["last_results"] = {"summary": results, "logs": logs_store, "f_str": f_str}

        # Results table
        st.subheader("📊 Results")
        df = pd.DataFrame(results)
        def fmt_root(v):
            return f"{v:.10g}" if isinstance(v, (int, float)) and not np.isnan(v) else "—"
        df_display = df.copy()
        df_display["Root Found"] = df_display["Root Found"].apply(fmt_root)
        st.table(df_display)

        # Best method
        succ = df[df["Status"] == "Success"]
        if not succ.empty:
            best = succ.sort_values("Iterations").iloc[0]
            st.success(f"Best: {best['Method']} ({int(best['Iterations'])} iterations)")
        else:
            st.info("No method achieved the requested tolerance.")

        # Function plot + iterates
        st.subheader("📈 Function plot + iterates")
        try:
            x_min = min(a, b) - 2
            x_max = max(a, b) + 2
            x_vals = np.linspace(x_min, x_max, 800)
            y_vals = []
            for xv in x_vals:
                try:
                    y_vals.append(f_func(xv))
                except Exception:
                    y_vals.append(np.nan)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x)"))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            for m, logs in logs_store.items():
                if not logs:
                    continue
                xs = []
                ys = []
                for ln in logs:
                    if "c" in ln:
                        xs.append(ln.get("c")); ys.append(ln.get("f(c)"))
                    elif "x" in ln and "f(x)" in ln:
                        xs.append(ln.get("x")); ys.append(ln.get("f(x)"))
                    elif "x1" in ln:
                        xs.append(ln.get("x1")); ys.append(ln.get("f(x1)"))
                if xs:
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+lines", name=f"{m} iterates", line=dict(dash="dot")))

            fig.update_layout(title=f"f(x) = {html.escape(f_str)}", xaxis_title="x", yaxis_title="f(x)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plot error: {e}")

        # Convergence plot
        st.subheader("📉 Convergence (|residual| vs iteration)")
        conv_fig = go.Figure()
        any_conv = False
        for m, logs in logs_store.items():
            if not logs:
                continue
            any_conv = True
            iters = []
            residuals = []
            for ln in logs:
                iters.append(ln.get("iter"))
                res = None
                if "f(c)" in ln and ln.get("f(c)") is not None:
                    res = abs(ln.get("f(c)"))
                elif "f(x)" in ln and ln.get("f(x)") is not None:
                    res = abs(ln.get("f(x)"))
                elif "f(x1)" in ln and ln.get("f(x1)") is not None:
                    res = abs(ln.get("f(x1)"))
                else:
                    res = np.nan
                residuals.append(res)
            conv_fig.add_trace(go.Scatter(x=iters, y=residuals, mode="lines+markers", name=m))
        if any_conv:
            conv_fig.update_layout(xaxis_title="Iteration", yaxis_title="|residual|", template="plotly_white")
            st.plotly_chart(conv_fig, use_container_width=True)
        else:
            st.info("No convergence logs available to plot.")

        # Logs and CSV download
        if show_logs:
            st.subheader("📜 Logs (expand per method)")
            for m, logs in logs_store.items():
                with st.expander(f"{m} — {len(logs)} steps"):
                    if not logs:
                        st.info("No logs for this method.")
                        continue
                    ldf = pd.DataFrame(logs)
                    st.dataframe(ldf)
                    csv_bytes = df_to_csv_bytes(ldf)
                    href = download_link_bytes(csv_bytes, f"{m.lower().replace(' ','_')}_log.csv", "Download CSV")
                    st.markdown(href, unsafe_allow_html=True)

# -------------------------
# Lagrange Mode
# -------------------------
else:
    st.header("📐 Lagrange Interpolation")

    left, right = st.columns([1, 1])
    with left:
        paste = st.text_area("Paste CSV rows (x,y) or leave blank")
        upload = st.file_uploader("Or upload CSV file (columns: x,y)", type=["csv"])
        if upload:
            try:
                df_csv = pd.read_csv(upload)
                if set(["x", "y"]).issubset(df_csv.columns):
                    st.session_state["lag_data"] = df_csv[["x", "y"]].dropna().to_dict(orient="list")
                    st.success("CSV loaded.")
                else:
                    st.error("CSV must contain 'x' and 'y' columns.")
            except Exception as e:
                st.error(f"CSV read error: {e}")

        if paste.strip():
            try:
                rows = [r.strip() for r in paste.strip().splitlines() if r.strip()]
                xs = []; ys = []
                for r in rows:
                    parts = [p.strip() for p in r.split(",")]
                    if len(parts) >= 2:
                        xs.append(float(parts[0])); ys.append(float(parts[1]))
                if len(xs) >= 2:
                    st.session_state["lag_data"] = {"x": xs, "y": ys}
                    st.success("Data parsed from paste.")
                else:
                    st.error("Need at least two rows.")
            except Exception as e:
                st.error(f"Parse error: {e}")

        st.markdown("---")
        x_manual = st.text_input("X points (comma separated)", value="5,6,9,11")
        y_manual = st.text_input("Y points (comma separated)", value="12,13,14,16")
        target_x = st.number_input("Target X", value=10.0)
        calc_btn = st.button("Calculate Interpolation")

    with right:
        st.markdown("Preview / Result")
        if "lag_data" in st.session_state and st.session_state["lag_data"]:
            data = st.session_state["lag_data"]
            if isinstance(data, dict) and "x" in data and "y" in data:
                pts_df = pd.DataFrame({"x": data["x"], "y": data["y"]})
                st.table(pts_df)
        else:
            st.info("No data loaded yet. Paste, upload, or use manual input then Calculate.")

    if calc_btn:
        try:
            xs = [float(k.strip()) for k in x_manual.split(",") if k.strip()]
            ys = [float(k.strip()) for k in y_manual.split(",") if k.strip()]
            if len(xs) != len(ys):
                st.error("X and Y must have same length.")
            elif len(xs) < 2:
                st.error("Need at least two points.")
            else:
                y_res = lagrange_interpolate(xs, ys, float(target_x))
                if y_res is None:
                    st.error("Error in points (repeated x?).")
                else:
                    st.success(f"Interpolated: y({target_x}) = {y_res:.6g}")
                    st.session_state["lag_data"] = {"x": xs, "y": ys}
        except Exception as e:
            st.error(f"Input error: {e}")

    if "lag_data" in st.session_state and st.session_state["lag_data"]:
        dd = st.session_state["lag_data"]
        if isinstance(dd, dict) and "x" in dd and "y" in dd:
            xs = dd["x"]; ys = dd["y"]
            x_s = np.linspace(min(xs), max(xs), 400)
            y_s = [lagrange_interpolate(xs, ys, xv) for xv in x_s]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_s, y=y_s, mode="lines", name="Lagrange curve"))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Data points"))
            st.plotly_chart(fig, use_container_width=True)
            pts_df = pd.DataFrame({"x": xs, "y": ys})
            csv_bytes = df_to_csv_bytes(pts_df)
            href = download_link_bytes(csv_bytes, "lagrange_points.csv", "Download points CSV")
            st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("NumComp Pro (Option B) — safer parsing, logs, exports. Verify results for critical tasks.")
