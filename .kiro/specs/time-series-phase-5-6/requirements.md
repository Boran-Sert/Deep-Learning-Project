# Requirements Document

## Introduction

This document specifies the requirements for Phase 5 (Event-Driven Reporting, Statistics, and Explainability) and Phase 6 (Visualization Layer) of the Time Series Anomaly Detection project. These phases enhance the existing anomaly detection system by adding event-driven architecture for reporting, statistical analysis capabilities, explainability features for automata decisions, and a comprehensive visualization layer.

## Glossary

- **System**: Time Series Anomaly Detection System
- **Event**: A notification object that signals significant occurrences in the system
- **Observer Pattern**: A design pattern where objects (observers) subscribe to events and react when they are published
- **Event Bus**: A central communication channel for publishing and subscribing to events
- **ReportManager**: An observer component that listens to events and computes statistics across experiments
- **StatisticalAnalyzer**: A component that performs statistical significance tests on model performance metrics
- **Wilcoxon Test**: A non-parametric statistical test for comparing paired samples
- **McNemar Test**: A statistical test for comparing paired proportions (e.g., classification accuracy)
- **ExplainabilityEngine**: A component that generates detailed explanations for automata decisions
- **Counterfactual Analysis**: Analysis of what changes would flip a decision from anomaly to normal or vice versa
- **VisualizationManager**: A component that generates and saves plots and figures for analysis
- **ModelTrainedEvent**: An event fired when a model completes training
- **EvaluationCompletedEvent**: An event fired when an evaluation scenario completes
- **AutomataDecisionEvent**: An event fired during unseen pattern handling in the automata model
- **SensitivityAnalysisEvent**: An event fired after parameter sensitivity analysis completes
- **F1 Score**: Harmonic mean of precision and recall
- **ROC Curve**: Receiver Operating Characteristic curve plotting TPR vs FPR
- **PR Curve**: Precision-Recall curve
- **Transition Probability**: Probability of transitioning from one automata state to another

## Requirements

### Requirement 1: Event-Driven Architecture

**User Story:** As a developer, I want to implement an event-driven architecture, so that different components can communicate without tight coupling and reporting can be triggered automatically by system events.

#### Acceptance Criteria

1. WHEN a model completes training, THE EventBus SHALL publish a ModelTrainedEvent containing model_name, dataset_name, seed, fold_idx, and training_duration_seconds
2. WHEN an evaluation scenario completes, THE EventBus SHALL publish an EvaluationCompletedEvent containing scenario, train_dataset, test_dataset, model_name, seed, metrics, fold_idx, and note
3. WHEN the automata model processes unseen patterns, THE EventBus SHALL publish an AutomataDecisionEvent containing scenario, unseen_rate, mapping_log, metrics, fold_idx, and note
4. WHEN parameter sensitivity analysis completes, THE EventBus SHALL publish a SensitivityAnalysisEvent containing results array with sensitivity analysis data
5. THE EventBus SHALL support the Observer pattern with subscribe and publish methods
6. WHERE multiple observers exist, THE EventBus SHALL notify all subscribed observers when an event is published

### Requirement 2: ReportManager Observer

**User Story:** As a data scientist, I want the ReportManager to listen to events and compute statistics, so that I can analyze model performance across multiple seeds and folds.

#### Acceptance Criteria

1. WHEN a ModelTrainedEvent is received, THE ReportManager SHALL record the training duration for the model
2. WHEN an EvaluationCompletedEvent is received, THE ReportManager SHALL extract and store metrics (F1, Accuracy, Precision, Recall) for the model
3. WHERE multiple seeds are used, THE ReportManager SHALL compute mean and standard deviation of F1, Accuracy, Precision, and Recall across all seeds
4. WHERE multiple folds are used, THE ReportManager SHALL compute mean and standard deviation of metrics across all folds
5. THE ReportManager SHALL organize statistics in Table 5 format with columns for Model, Seed, Fold, F1, Accuracy, Precision, Recall, Training Time, and Inference Time
6. WHERE a SensitivityAnalysisEvent is received, THE ReportManager SHALL store parameter sensitivity results for later analysis

### Requirement 3: StatisticalAnalyzer

**User Story:** As a researcher, I want to perform statistical significance tests on model comparisons, so that I can determine if performance differences are statistically significant.

#### Acceptance Criteria

1. WHEN two models are compared on the same dataset, THE StatisticalAnalyzer SHALL implement the Wilcoxon signed-rank test for paired samples
2. WHEN comparing classification accuracy between two models, THE StatisticalAnalyzer SHALL implement the McNemar test for paired proportions
3. WHERE statistical tests are performed, THE StatisticalAnalyzer SHALL return p-values and test statistics
4. WHERE p-values are returned, THE StatisticalAnalyzer SHALL indicate statistical significance at α=0.05 level
5. THE StatisticalAnalyzer SHALL support comparison of F1 scores, Accuracy, Precision, and Recall metrics
6. WHERE multiple comparisons are made, THE StatisticalAnalyzer SHALL apply appropriate multiple testing corrections

### Requirement 4: ExplainabilityEngine JSON Output

**User Story:** As a system operator, I want detailed JSON output for each automata decision step, so that I can understand and verify the decision-making process.

#### Acceptance Criteria

1. WHEN an automata decision is made, THE ExplainabilityEngine SHALL generate a JSON object containing time_step, state, pattern, status, mapped_to, probability, decision, and Confidence Score
2. WHERE a pattern is in the vocabulary, THE ExplainabilityEngine SHALL set status to "known" and include the transition probability
3. WHERE a pattern is not in the vocabulary, THE ExplainabilityEngine SHALL set status to "unseen" and include the mapped_to field with the closest matching pattern
4. WHERE a decision is made, THE ExplainabilityEngine SHALL include the probability of the observed path under the normal model
5. WHERE a decision is anomaly, THE ExplainabilityEngine SHALL set decision to 1 and include a confidence score based on path probability
6. WHERE a decision is normal, THE ExplainabilityEngine SHALL set decision to 0 and include a confidence score based on path probability

### Requirement 5: Counterfactual Analysis Module

**User Story:** As a data scientist, I want counterfactual analysis for automata decisions, so that I can understand what changes would flip an anomaly detection decision.

#### Acceptance Criteria

1. WHEN an anomaly decision is made, THE CounterfactualAnalyzer SHALL identify which pattern transitions contributed most to the anomaly score
2. WHERE pattern transitions are identified, THE CounterfactualAnalyzer SHALL suggest alternative transitions that would reduce the anomaly score
3. THE CounterfactualAnalyzer SHALL generate counterfactual explanations showing what pattern changes would result in a normal decision
4. WHERE counterfactuals are generated, THE CounterfactualAnalyzer SHALL include the probability of the counterfactual path
5. WHERE counterfactuals are generated, THE CounterfactualAnalyzer SHALL compute the difference in anomaly score between original and counterfactual
6. THE CounterfactualAnalyzer SHALL output counterfactual explanations in JSON format with original_pattern, counterfactual_pattern, original_probability, counterfactual_probability, and score_difference

### Requirement 6: VisualizationManager - General

**User Story:** As a user, I want a VisualizationManager that generates and saves plots, so that I can visually analyze model performance and system behavior.

#### Acceptance Criteria

1. THE VisualizationManager SHALL use Matplotlib and Seaborn for plotting
2. WHERE figures are generated, THE VisualizationManager SHALL save them to the reports/figures/ directory
3. WHERE a figure is saved, THE VisualizationManager SHALL use descriptive filenames based on the plot type and experiment parameters
4. THE VisualizationManager SHALL support saving figures in PNG and PDF formats
5. WHERE multiple seeds or folds are involved, THE VisualizationManager SHALL create summary plots aggregating across runs

### Requirement 7: Confusion Matrix Visualization

**User Story:** As a data scientist, I want confusion matrix plots, so that I can visualize classification performance.

#### Acceptance Criteria

1. WHERE evaluation metrics are available, THE VisualizationManager SHALL generate a confusion matrix plot
2. THE Confusion Matrix plot SHALL show True Positives, True Negatives, False Positives, and False Negatives
3. WHERE multiple models are compared, THE VisualizationManager SHALL create side-by-side confusion matrices
4. WHERE multiple seeds are used, THE VisualizationManager SHALL create a summary confusion matrix aggregated across seeds
5. THE Confusion Matrix plot SHALL include percentage annotations and color coding

### Requirement 8: ROC/PR Curve Visualization

**User Story:** As a data scientist, I want ROC and Precision-Recall curves, so that I can analyze model performance across different thresholds.

#### Acceptance Criteria

1. WHERE prediction scores are available, THE VisualizationManager SHALL generate an ROC curve plot
2. WHERE prediction scores are available, THE VisualizationManager SHALL generate a Precision-Recall curve plot
3. WHERE multiple models are compared, THE VisualizationManager SHALL plot multiple curves on the same figure with a legend
4. WHERE multiple seeds are used, THE VisualizationManager SHALL compute mean and confidence intervals across seeds
5. THE ROC curve SHALL include the AUC score in the legend
6. THE PR curve SHALL include the average precision score in the legend

### Requirement 9: Automata State Diagram

**User Story:** As a researcher, I want an automata state diagram visualization, so that I can understand the structure of the learned automata model.

#### Acceptance Criteria

1. WHERE the automata transition matrix is available, THE VisualizationManager SHALL generate a state diagram using Graphviz
2. THE State Diagram SHALL represent states as nodes and transitions as directed edges
3. WHERE transition probabilities are available, THE State Diagram SHALL label edges with probability values
4. WHERE states have different meanings (normal vs. anomaly-prone), THE State Diagram SHALL use different node colors
5. THE State Diagram SHALL be saved in both PNG and PDF formats
6. WHERE the vocabulary is large, THE VisualizationManager SHALL provide an option to show only the most probable transitions

### Requirement 10: Transition Probability Heatmap

**User Story:** As a data scientist, I want a transition probability heatmap, so that I can visualize the probability distribution of automata transitions.

#### Acceptance Criteria

1. WHERE the automata transition matrix is available, THE VisualizationManager SHALL generate a transition probability heatmap
2. THE Heatmap SHALL use a color gradient to represent transition probabilities (low probability = cool colors, high probability = warm colors)
3. WHERE states are labeled, THE Heatmap SHALL show state names on both axes
4. WHERE the vocabulary is large, THE VisualizationManager SHALL provide an option to show only the top N most probable transitions
5. THE Heatmap SHALL include a colorbar legend
6. THE Heatmap SHALL be saved in both PNG and PDF formats

### Requirement 11: Parameter Sensitivity Plots

**User Story:** As a researcher, I want parameter sensitivity plots, so that I can understand how automata parameters affect performance.

#### Acceptance Criteria

1. WHERE sensitivity analysis results are available, THE VisualizationManager SHALL generate parameter sensitivity plots
2. WHERE window_size and alphabet_size are varied, THE VisualizationManager SHALL create a 2D heatmap of performance metrics
3. WHERE multiple metrics are available, THE VisualizationManager SHALL create separate plots for F1, Accuracy, Precision, and Recall
4. THE Sensitivity Plots SHALL include annotations showing the optimal parameter combination
5. WHERE state count and transition density are available, THE VisualizationManager SHALL create additional plots showing these metrics
6. THE Sensitivity Plots SHALL be saved in both PNG and PDF formats

### Requirement 12: Configuration Integration

**User Story:** As a developer, I want all components to use centralized configuration, so that the system remains maintainable and consistent.

#### Acceptance Criteria

1. WHERE a component needs configuration, THE Component SHALL use ConfigurationManager to retrieve settings
2. WHERE file paths are needed, THE Component SHALL use paths from the configuration file
3. WHERE default values are needed, THE Component SHALL use sensible defaults specified in the configuration
4. THE Configuration SHALL include settings for visualization output directory, figure formats, and statistical test parameters
5. WHERE event-driven components are used, THE Configuration SHALL include settings for event filtering and logging

### Requirement 13: Data Leakage Prevention

**User Story:** As a developer, I want to ensure data leakage is prevented in reporting and visualization, so that performance estimates remain valid.

#### Acceptance Criteria

1. WHERE statistics are computed, THE ReportManager SHALL only use metrics from test sets, not training sets
2. WHERE visualizations are generated, THE VisualizationManager SHALL use only test set predictions and ground truth
3. WHERE statistical tests are performed, THE StatisticalAnalyzer SHALL use only paired test set results
4. WHERE counterfactual analysis is performed, THE CounterfactualAnalyzer SHALL use only test set data
5. WHERE sensitivity analysis is performed, THE SensitivityAnalyzer SHALL use only test set results

### Requirement 14: SOLID and OOP Principles

**User Story:** As a developer, I want to follow SOLID and OOP principles, so that the code remains maintainable and extensible.

#### Acceptance Criteria

1. WHERE new event types are needed, THE System SHALL allow adding new event classes without modifying existing code
2. WHERE new observers are needed, THE System SHALL allow adding new observer classes without modifying existing code
3. WHERE new statistical tests are needed, THE System SHALL use an interface-based approach for statistical tests
4. WHERE new visualizations are needed, THE System SHALL use an interface-based approach for visualization components
5. WHERE components depend on each other, THE System SHALL use dependency injection through the EventBus
6. THE System SHALL follow the Single Responsibility Principle with separate classes for reporting, statistics, explainability, and visualization
