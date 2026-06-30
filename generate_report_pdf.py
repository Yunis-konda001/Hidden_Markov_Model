"""I generate the PDF report from markdown content and figures."""
import os
from fpdf import FPDF

REPORT_DIR = "report"
FIG_DIR = "figures"
OUTPUT = os.path.join(REPORT_DIR, "formative2_hmm_report.pdf")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "Formative 2: HMM Activity Recognition", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 7, title)
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_figure(self, path, caption, w=170):
        if os.path.exists(path):
            self.image(path, w=w)
            self.set_font("Helvetica", "I", 9)
            self.multi_cell(0, 5, caption)
            self.ln(3)


def main():
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.section_title("1. Background and Motivation")
    pdf.body_text(
        "I built a smartphone activity monitor to detect standing, walking, jumping, and still "
        "states from accelerometer and gyroscope data. This supports fitness tracking and sedentary "
        "reminders without extra hardware. The true activity is hidden behind noisy sensors, so I "
        "used a Hidden Markov Model to infer activity states over time."
    )

    pdf.section_title("2. Data Collection and Preprocessing")
    pdf.body_text(
        "I used Sensor Logger to record 10 sessions per activity (standing, walking, jumping, still). "
        "Each recording lasted 5-10 seconds. Files are stored in data/ with accel and gyro CSV pairs. "
        "My sampling rate is about 99.9 Hz on one phone, so I did not resample. I used 1-second "
        "windows with 0.5-second steps (99 samples per window at 100 Hz). walking_07_accel.csv was "
        "missing and was excluded from training."
    )
    pdf.add_figure(os.path.join(FIG_DIR, "raw_sensor_samples.png"), "Figure 1: Raw accelerometer magnitude per activity.")

    pdf.section_title("3. Feature Extraction")
    pdf.body_text(
        "I extracted 8 features per window: mean_mag, std_mag, var_mag, sma, corr_xy (time domain), "
        "and dom_freq, spec_energy, fft_peak (FFT on accel magnitude). I applied z-score normalization "
        "using training statistics only so large features do not dominate the HMM."
    )

    pdf.add_page()
    pdf.section_title("4. HMM Setup and Implementation")
    pdf.body_text(
        "I used a 4-state Gaussian HMM (hmmlearn). Hidden states map to the four activities. "
        "Observations are 8D feature vectors. Baum-Welch training used tol=1e-4 and converged in "
        "2 iterations. Viterbi decoding (model.predict) finds the most likely state sequence. "
        "I trained on recordings 01-08 and tested on unseen recordings 09-10."
    )
    pdf.add_figure(os.path.join(FIG_DIR, "baum_welch_convergence.png"), "Figure 2: Baum-Welch log-likelihood convergence.")
    pdf.add_figure(os.path.join(FIG_DIR, "transition_matrix.png"), "Figure 3: Transition probability matrix (A).")
    pdf.add_figure(os.path.join(FIG_DIR, "emission_means.png"), "Figure 4: Emission means per activity state (B).")

    pdf.add_page()
    pdf.section_title("5. Results and Interpretation")
    pdf.body_text(
        "On unseen test data (recordings 09 and 10), the model achieved 99.6% window-level accuracy. "
        "Jumping and still were easiest to separate. Standing was hardest and had one misclassified window."
    )
    pdf.add_figure(os.path.join(FIG_DIR, "decoded_sequence.png"), "Figure 5: Viterbi decoded sequence on unseen walking_09.")
    pdf.add_figure(os.path.join(FIG_DIR, "confusion_matrix.png"), "Figure 6: Confusion matrix on unseen test windows.")

    pdf.section_title("Evaluation Metrics (Unseen Test Data)")
    pdf.set_font("Courier", "", 9)
    table = (
        "Activity   Samples  Sensitivity  Specificity  Accuracy\n"
        "standing      30      0.967        1.000      0.992\n"
        "walking       33      1.000        0.989      0.992\n"
        "jumping       33      1.000        1.000      1.000\n"
        "still         32      1.000        1.000      1.000\n"
    )
    pdf.multi_cell(0, 4, table)
    pdf.ln(4)

    pdf.section_title("6. Discussion and Conclusion")
    pdf.body_text(
        "Transition probabilities show high self-loop values, which matches short clips where activity "
        "rarely changes. Sensor noise had limited impact because features average over 1-second windows. "
        "Improvements: re-record missing walking_07, collect more data, add sensors, and test in new "
        "environments."
    )

    pdf.section_title("7. Task Allocation (Individual Work)")
    pdf.body_text(
        "I completed all tasks: data collection, cleaning, feature extraction, HMM training, evaluation, "
        "visualizations, and this report."
    )

    pdf.output(OUTPUT)
    print(f"I saved the report to {OUTPUT}")


if __name__ == "__main__":
    main()
