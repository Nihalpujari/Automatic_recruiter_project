AI-Driven Technical Talent Identification System
1. Project Overview
This project uses Machine Learning (ML) pipelines to analyze developer activity across three major platforms: StackOverflow, Kaggle, and GitHub. The goal is to move beyond "vanity metrics" (like total followers or raw upvotes) and instead evaluate candidates based on Role-Specific Relationships—identifying balance between quality, consistency, and scale.
The system outputs the Top 3 Candidates for three specific roles:
Developer (Execution & Documentation focus)
Senior Developer (Authority & Impact focus)
Solution Architect (Scale & System Breadth focus)
2. Dataset Definitions
A. StackOverflow (stackoverflow_200.csv)
Metric Focus: Problem-solving accuracy and community validation.
Key Columns Used: reputation, accepted_answer_ratio, avg_score_per_answer, top_tags.
B. Kaggle (kaggle-preprocessed.csv)
Metric Focus: Project portfolio quality and data engineering capability.
Key Columns Used: Usability, Upvotes, Medals, No_of_files, size.
C. GitHub (github_candidates_1.csv)
Metric Focus: Coding velocity, community adoption, and engineering rigor.
Key Columns Used: commits_12m, total_stars, total_forks, has_cicd.
3. Machine Learning Workflow
Step 1: Data Cleaning & Noise Reduction
Standardization: Kaggle "Dataset Size" was converted from mixed strings (KB, MB, GB) into a unified float (MB) to allow mathematical comparison.
Missing Values: Used Median Imputation for technical scores and categorized missing Medals as "No Medal" to maintain index continuity.
Aggregation: For Kaggle and GitHub, the model groups repeated usernames/authors to evaluate a candidate’s cumulative portfolio rather than a single viral entry.
Step 2: Handling Outliers
Winsorization: For StackOverflow Reputation, values were capped at the 95th percentile. This ensures that extreme outliers (users with millions of points) don't "blind" the model to high-potential candidates.
Logarithmic Scaling: For Upvotes and Stars, we applied log1p transformation. This emphasizes the growth of impact rather than just raw volume.
Step 3: Feature Engineering
Tag Breadth: Created an index for the number of unique technology domains a candidate has mastered.
Medal Weighting: Mapped categorical Medals to a numeric weight (Gold=5, Silver=3, Bronze=1) to quantify industry validation.
Step 4: Min-Max Scaling
All variables are normalized to a 0.0 - 1.0 range. This allows us to compare different units (e.g., comparing a 9.4/10 Usability score with a 200,000 Reputation count) on a fair, weighted scale.
4. Selection Basis by Job Profile
Role: Developer
Basis: Efficiency & Maintainability.
Logic: The model looks for the relationship between Accepted Answer Ratio (does their code work?) and Usability Mean (is their work well-documented?).
Ideal Candidate: Someone who solves problems correctly the first time and hands over clean, documented projects.
Role: Senior Developer
Basis: Authority & Peer Validation.
Logic: The model evaluates the relationship between Reputation (long-term trust) and Community Impact (Medals and Upvotes).
Ideal Candidate: An industry veteran whose technical choices have been vetted and approved by thousands of other high-level engineers.
Role: Solution Architect
Basis: System Complexity & Scale.
Logic: The model prioritizes Tag Breadth (domain diversity) and the Scale Index (handling massive file counts and data volumes).
Ideal Candidate: Someone who understands how broad systems interact and has experience managing high-entropy, large-scale data environments (e.g., managing 27,000+ files or 50GB+ systems).
5. How to Run the Model
Environment: Ensure you have python, pandas, numpy, and scikit-learn installed.
Display Settings: The code includes pd.set_option to prevent terminal truncation (...) for large data outputs.
Execution: Run the specific script for the platform you wish to analyze (Kaggle-only or Unified).
Output: The system will print the Top 3 candidates and a generated Basis statement explaining the metrics behind their selection.
6. Selection Criteria Matrix
Metric	Developer	Senior Developer	Solution Architect
Primary	Accepted Ratio / Usability	Reputation / Medals	Tag Breadth / Data Scale
Secondary	Coding Velocity	Avg. Answer Score	CI/CD Usage / Forks
Behavior	Accuracy & Documentation	Leadership & Impact	Strategy & Complexity
Final Note for Hiring Managers
Candidates selected by this model rank in the top percentiles of balanced performance. They are prioritized not just for having "big numbers," but for maintaining a high quality-to-quantity ratio across their entire technical history.