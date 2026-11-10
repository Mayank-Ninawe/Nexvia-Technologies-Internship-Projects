"""Execute notebook cells in-process and persist outputs into the notebook file."""

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
import nbformat as nbf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(".").resolve()))
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/Temp")


class TeeStream(io.StringIO):
    def __init__(self, target_stream):
        super().__init__()
        self.target = target_stream

    def write(self, s):
        super().write(s)
        self.target.write(s)
        self.target.flush()

    def flush(self):
        super().flush()
        self.target.flush()


def run_notebook(notebook_path: Path):
    print("\n==========================================")
    print(f"Executing: {notebook_path.name}")
    print("==========================================")

    nb = nbf.read(notebook_path, as_version=4)
    global_scope = {"__file__": str(notebook_path.resolve())}

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source
        cell.outputs = []
        cell.execution_count = idx + 1

        stdout_buf = TeeStream(sys.__stdout__)
        stderr_buf = TeeStream(sys.__stderr__)

        print(f"[{notebook_path.name}] Executing Cell {idx}...")

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(source, global_scope)
        except Exception as e:
            err_text = f"{type(e).__name__}: {e}"
            print(f"Error in Cell {idx}: {err_text}")
            cell.outputs.append(
                nbf.v4.new_output(
                    output_type="error",
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=[err_text],
                )
            )
            nbf.write(nb, notebook_path)
            raise

        out_text = stdout_buf.getvalue()
        if out_text:
            cell.outputs.append(
                nbf.v4.new_output(output_type="stream", name="stdout", text=out_text)
            )
            print(out_text.strip())

        err_text = stderr_buf.getvalue()
        if err_text:
            cell.outputs.append(
                nbf.v4.new_output(output_type="stream", name="stderr", text=err_text)
            )

        # Check for any active matplotlib figures
        fig_nums = plt.get_fignums()
        if fig_nums:
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight")
                img_buf.seek(0)
                import base64
                b64_img = base64.b64encode(img_buf.read()).decode("utf-8")
                cell.outputs.append(
                    nbf.v4.new_output(
                        output_type="display_data",
                        data={"image/png": b64_img, "text/plain": "<Figure size ...>"},
                    )
                )
            plt.close("all")

    nbf.write(nb, notebook_path)
    print(f"Successfully finished and saved: {notebook_path.name}\n")


if __name__ == "__main__":
    notebook_name = sys.argv[1] if len(sys.argv) > 1 else None
    if notebook_name:
        run_notebook(Path(notebook_name))
    else:
        for nb_file in sorted(Path("ml/notebooks").glob("*.ipynb")):
            if nb_file.name.startswith("0"):
                run_notebook(nb_file)
