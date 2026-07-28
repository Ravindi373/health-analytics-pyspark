# Student Health & Biodata Analytics with PySpark

A small data engineering / analytics project that cleans a student biodata
dataset, engineers health-risk indicators, and produces department-level
health insights using PySpark.

## What it does

1. **Load** — reads `data/biodata_advanced.csv` into a Spark DataFrame and
   inspects its shape, schema, and first rows.
2. **Clean** — detects biologically invalid `0` values in `height_cm`,
   `weight_kg`, and `cholesterol_mg_dl`, treats them as missing, and
   imputes them using the mean for each `(gender, department)` group
   (falling back to the global mean if a group has no valid values).
3. **Feature engineering**
   - `bmi` = `weight_kg / (height_cm / 100)^2`
   - `bmi_category` — Underweight / Normal / Overweight / Obese
   - `high_bp` — `True` if systolic ≥ 140 or diastolic ≥ 90
   - `at_risk` — `True` if BMI category is Overweight/Obese, `high_bp` is
     `True`, or cholesterol ≥ 200
4. **Analysis**
   - Average BMI per department
   - Percentage of at-risk students per department
   - Mean systolic/diastolic BP by year
   - Pivot table: department × gender → mean BMI
   - Full sorted list of at-risk students
5. **Export** — writes the at-risk student list to
   `outputs/at_risk_students.csv`.

## Project structure

```
health-analytics-pyspark/
├── data/
│   └── biodata_advanced.csv       # raw dataset
├── outputs/
│   └── at_risk_students.csv       # generated on run
├── src/
│   └── health_analytics.py        # main PySpark script
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (VS Code)

1. **Prerequisites**: Python 3.9+ and Java 8/11/17 (PySpark needs a JVM).
   Check with:
   ```bash
   python --version
   java -version
   ```
2. Clone the repo and open it in VS Code:
   ```bash
   git clone <my-repo-url>
   cd health-analytics-pyspark
   code .
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. (In VS Code) select the `.venv` interpreter: **Ctrl/Cmd+Shift+P →
   Python: Select Interpreter → .venv**.

## Run

```bash
python src/health_analytics.py
```

This prints each stage of the analysis to the terminal and writes the
final at-risk student list to `outputs/at_risk_students.csv`.

## Findings

**Cleaning strategy** - Zero values in height, weight, and cholesterol are
physically impossible for a living person, so they were treated as missing
data rather than genuine measurements. Rather than dropping rows (which
would shrink an already small 100-row sample), missing values were imputed
using the mean for that student's `(gender, department)` group, since body
metrics vary meaningfully by gender and, to a lesser extent, by
department/lifestyle. Any group with no valid values at all falls back to
the global column mean.

**Department with the highest average BMI** - DS (Data Science), at
~25.6, narrowly ahead of IT (~25.0) and CS (~24.2). The gap is small
enough that it may simply reflect this dataset's sample rather than a
strong underlying pattern, but it's worth watching if it holds across a
larger sample.

**Department with the most at-risk students** - DS has the highest
at-risk rate (100% of its 30 students meet at least one risk criterion),
with IT close behind (~97%) and CS somewhat lower (~95%). Because the
at-risk definition is broad (any one of BMI, blood pressure, or
cholesterol), these percentages are high across the board.

**Highest-risk individual** - Student S018 (Kaveen, DS) has the highest
BMI in the dataset (~39.8, Obese) combined with elevated blood pressure
(150/87, meeting the high-BP threshold) and cholesterol above the 200
mg/dL threshold — all three risk factors present at once.

**Real-world use case** - A university health service could use this kind
of pipeline to flag students for a wellness check-in each semester,
prioritizing outreach to those with multiple simultaneous risk factors
rather than relying on any single measurement in isolation.

## Notes

- The dataset (`biodata_advanced.csv`) is a small, self-contained sample
  used to demonstrate the pipeline end-to-end.
- Built and tested with PySpark's local mode (`local[*]`), so no cluster
  or cloud setup is required to run it.
