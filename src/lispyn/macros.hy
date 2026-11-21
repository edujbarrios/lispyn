; Lispyn macros. Copyright 2026 Eduardo J. Barrios.
; SPDX-License-Identifier: Apache-2.0

(defmacro with-seed [seed #* body]
  "Seed Python, NumPy (when installed) and torch before evaluating BODY."
  `(do
    (import lispyn.reproducibility [seed-all])
    (seed-all ~seed)
    ~@body))

(defmacro train-for [model loader optimizer criterion epochs]
  "Run a standard PyTorch fit loop with explicit components."
  `(do
     (import lispyn.training [fit])
     (fit ~model ~loader ~optimizer :loss-fn ~criterion :epochs ~epochs)))

(defmacro with-eval [model #* body]
  "Evaluate with inference_mode and restore each child's train/eval flag."
  (setv model-symbol (hy.gensym "lispyn_model"))
  (setv states-symbol (hy.gensym "lispyn_states"))
  (setv child-symbol (hy.gensym "lispyn_child"))
  (setv flag-symbol (hy.gensym "lispyn_flag"))
  `(do
    (import torch)
    (setv ~model-symbol ~model)
    (setv ~states-symbol
      (list (map (fn [~child-symbol]
                   (tuple [~child-symbol (getattr ~child-symbol "training")]))
                 (.modules ~model-symbol))))
    (try
      (.eval ~model-symbol)
      (with [(torch.inference_mode)]
        ~@body)
      (finally
        (for [[~child-symbol ~flag-symbol] ~states-symbol]
          (setattr ~child-symbol "training" ~flag-symbol))))))

(defmacro curve [history #* options]
  "Generate a styled Matplotlib training curve from recorded metrics."
  `(do
    (import lispyn.plots [plot-history])
    (plot-history ~history ~@options)))

(defmacro defcurve [name #* options]
  "Define a reusable curve design; its arguments become default plot options."
  `(do
    (import lispyn.plots [make-plotter])
    (setv ~name (make-plotter ~@options))))

(defmacro md [#* content]
  "Document a notebook cell; Lispyn turns literal top-level calls into Markdown."
  None)

(defmacro markdown [#* content]
  "Long-form alias for md."
  None)

(defmacro head-tune [model data head #* options]
  "Train only named model head using the native PyTorch optimizer."
  `(do
    (import lispyn.workflow [train])
    (train ~model ~data :head ~head ~@options)))

(defmacro full-tune [model data #* options]
  "Unfreeze all layers, then fine-tune with a fresh native optimizer."
  (setv model-symbol (hy.gensym "lispyn_model"))
  `(do
    (import lispyn.training [trainable])
    (import lispyn.workflow [train])
    (setv ~model-symbol ~model)
    (trainable ~model-symbol)
    (train ~model-symbol ~data ~@options)))
(defmacro ensure-package [requirement #* options]
  "Install missing named pip requirements when this cell is executed."
  `(do
    (import lispyn.dependencies)
    (lispyn.dependencies.ensure-package ~requirement ~@options)))

(defmacro use-package [requirement #* options]
  "Install a missing module if needed, then return it for ordinary Hy use."
  `(do
    (import lispyn.dependencies)
    (lispyn.dependencies.use-package ~requirement ~@options)))

(defmacro pip-install [#* requirements]
  "Generate an independent dependency cell; install only when executed."
  `(do
    (import lispyn.dependencies [ensure-packages])
    (ensure-packages ~@requirements)))

(defmacro compare-curves [runs #* options]
  "Compare matching metrics from multiple named training runs in one chart."
  `(do
    (import lispyn.plots [plot-comparison])
    (plot-comparison ~runs ~@options)))

(defmacro confusion-matrix [actual predicted #* options]
  "Plot classification labels as a styled confusion matrix."
  `(do
    (import lispyn.classification [plot-confusion-matrix])
    (plot-confusion-matrix ~actual ~predicted ~@options)))

(defmacro defconfusion [name #* options]
  "Create a reusable confusion-matrix design with default options."
  `(do
    (import lispyn.classification [make-confusion-plotter])
    (setv ~name (make-confusion-plotter ~@options))))

(defmacro train-until [model data patience #* options]
  "Train up to :epochs, stopping after PATIENCE epochs without improvement."
  `(do
    (import lispyn.workflow [train])
    (train ~model ~data :patience ~patience ~@options)))

(defmacro raw-cell [#* content]
  "Emit a raw Jupyter cell from literal strings; no-op during Hy execution."
  None)

(defmacro cell-tags [#* tags]
  "Attach literal Jupyter metadata tags to the next code cell."
  None)

(defmacro classification-report [actual predicted #* options]
  "Compute precision, recall, F1 and support without scikit-learn."
  `(do
    (import lispyn.classification)
    (lispyn.classification.classification_report ~actual ~predicted ~@options)))

(defmacro train-accumulating [model data steps #* options]
  "Train with STEPS microbatches per optimizer update; forwards PyTorch options."
  `(do
    (import lispyn.workflow [train])
    (train ~model ~data :accumulation-steps ~steps ~@options)))

(defmacro learning-rate-curve [history #* options]
  "Visualize a native PyTorch learning-rate scheduler as a Matplotlib figure."
  `(do
    (import lispyn.plots [plot-learning-rate])
    (plot-learning-rate ~history ~@options)))

(defmacro train-val-test [dataset #* options]
  "Create train, validation and test PyTorch Subsets with no data leakage."
  `(do
    (import lispyn.data [split-dataset-three])
    (split-dataset-three ~dataset ~@options)))

(defmacro cross-validation-folds [dataset #* options]
  "Yield seeded, optionally stratified or grouped PyTorch k-fold Subsets."
  `(do
    (import lispyn.data [k-fold-dataset])
    (k-fold-dataset ~dataset ~@options)))


; Cross-validation workflows are intentionally authored through Hy macros.

(defmacro tabular-train-val-test [frame #* options]
  "Split a pandas DataFrame into seeded train/validation/test partitions."
  `(do
    (import lispyn.data [train-validation-test-split])
    (train-validation-test-split ~frame ~@options)))

(defmacro tabular-cross-validation-folds [frame #* options]
  "Yield pandas (train, validation) folds with class/group isolation."
  `(do
    (import lispyn.data [k-fold-split])
    (k-fold-split ~frame ~@options)))

(defmacro repeated-cross-validation-folds [dataset #* options]
  "Yield (repeat, fold, train_subset, val_subset) for repeated k-fold."
  `(do
    (import lispyn.cv_workflows [repeated-k-fold-dataset])
    (repeated-k-fold-dataset ~dataset ~@options)))

(defmacro repeated-tabular-folds [frame #* options]
  "Yield (repeat, fold, train_frame, val_frame) for repeated k-fold."
  `(do
    (import lispyn.cv_workflows [repeated-k-fold-split])
    (repeated-k-fold-split ~frame ~@options)))

(defmacro cross-validate [dataset scorer #* options]
  "Run a Hy fold-scoring callback on fresh PyTorch CV partitions."
  `(do
    (import lispyn.cv_workflows [run-dataset-cross-validation])
    (run-dataset-cross-validation ~dataset ~scorer ~@options)))

(defmacro cross-validate-table [frame scorer #* options]
  "Run a Hy callback on fresh, optionally grouped pandas CV partitions."
  `(do
    (import lispyn.cv_workflows [run-frame-cross-validation])
    (run-frame-cross-validation ~frame ~scorer ~@options)))

(defmacro save-cross-validation [report path]
  "Save a privacy-minimal fold-score summary to a JSON file."
  `(do
    (import lispyn.cv_workflows [save-cv-report])
    (save-cv-report ~report ~path)))

; Temporal workflows preserve order, with optional positional embargo gaps.

(defmacro temporal-train-val-test [dataset #* options]
  "Chronological train/validation/test PyTorch Subsets with optional gaps."
  `(do
    (import lispyn.temporal [temporal-three-way-dataset])
    (temporal-three-way-dataset ~dataset ~@options)))

(defmacro temporal-folds [dataset #* options]
  "Yield expanding or rolling chronological PyTorch training/validation folds."
  `(do
    (import lispyn.temporal [temporal-dataset-folds])
    (temporal-dataset-folds ~dataset ~@options)))

(defmacro temporal-table-split [frame #* options]
  "Chronologically split a pandas table; :time-column sorts unique timestamps."
  `(do
    (import lispyn.temporal [temporal-three-way-frame])
    (temporal-three-way-frame ~frame ~@options)))

(defmacro temporal-table-folds [frame #* options]
  "Yield ordered pandas time-series folds with an optional embargo gap."
  `(do
    (import lispyn.temporal [temporal-frame-folds])
    (temporal-frame-folds ~frame ~@options)))

(defmacro cross-validate-temporal [dataset scorer #* options]
  "Run a fresh Hy fold-scoring callback on walk-forward PyTorch windows."
  `(do
    (import lispyn.temporal [run-temporal-dataset-cv])
    (run-temporal-dataset-cv ~dataset ~scorer ~@options)))

(defmacro cross-validate-temporal-table [frame scorer #* options]
  "Run a fresh Hy fold-scoring callback on ordered pandas time windows."
  `(do
    (import lispyn.temporal [run-temporal-frame-cv])
    (run-temporal-frame-cv ~frame ~scorer ~@options)))


; Experiment storytelling: declarative notebook parameters and rich reports.

(defmacro experiment-params [#* assignments]
  "Declare notebook parameters via (setv name value) forms.

  Lispyn builds a tagged Jupyter parameters code cell. In ordinary Hy
  execution, the forms simply assign their variables.
  "
  `(do ~@assignments))

(defmacro report-run [run #* options]
  "Render and optionally save an explanatory Markdown training summary.

  In Jupyter this displays rich Markdown; in terminal Hy it prints text.
  Supports a workflow.Run or a mapping of metric histories.
  "
  `(do
    (import lispyn.reporting [show-run-report])
    (show-run-report ~run ~@options)))


; Evidence-backed experiment narratives in ordinary Hy.

(defmacro experiment-check [label condition #* options]
  "Show an explicit, testable evidence claim; optionally fail in strict mode."
  `(do
    (import lispyn.evidence [evaluate-check])
    (evaluate-check ~label ~condition ~@options)))

(defmacro checklist-report [checks #* options]
  "Show a reproducible summary of explicit Hy experiment checks."
  `(do
    (import lispyn.evidence [summarize-evidence])
    (summarize-evidence ~checks ~@options)))

; Live, human-readable experiment results written entirely in Hy.

(defmacro narrate [template #* options]
  "Render metric-aware Markdown with explicit named scalars at runtime."
  `(do
    (import lispyn.narrative [show-narrative])
    (show-narrative ~template ~@options)))

(defmacro metric-table [metrics #* options]
  "Display an explicit table of finite scalar metrics in Jupyter or CLI."
  `(do
    (import lispyn.narrative [show-metric-summary])
    (show-metric-summary ~metrics ~@options)))

; Turn saved native PyTorch runs into a readable, ranked research notebook.

(defmacro experiment-leaderboard [directory #* options]
  "Rank recorded experiment histories as Jupyter Markdown or terminal text."
  `(do
    (import lispyn.leaderboard [show-leaderboard])
    (show-leaderboard ~directory ~@options)))

; Compare saved PyTorch research runs directly within a Hy/Jupyter experiment.

(defmacro compare-runs [runs #* options]
  "Show a shareable baseline-vs-candidate Markdown table from saved histories."
  `(do
    (import lispyn.research_comparison [show-run-comparison])
    (show-run-comparison ~runs ~@options)))

(defmacro plot-runs [runs #* options]
  "Visualize a selected recorded metric across saved runs as a Matplotlib plot."
  `(do
    (import lispyn.research_comparison [plot-run-comparison])
    (plot-run-comparison ~runs ~@options)))

; Small experiment diagnostics without a required PyTorch installation.

(defmacro assert-close [actual expected #* options]
  "Assert that two finite numeric scalars match within explicit tolerances."
  `(do
    (import lispyn.utilities)
    (lispyn.utilities.assert-close ~actual ~expected ~@options)))

(defmacro with-timing [label #* body]
  "Time a Hy body and print elapsed seconds, including on errors."
  `(do
    (import lispyn.utilities)
    (with [(lispyn.utilities.timing ~label)]
      ~@body)))

; Aggregate data-quality diagnostics for NumPy arrays and pandas tables.

(defmacro array-profile [values #* options]
  "Display NumPy shape, NaNs, infinities and finite-only summary statistics."
  `(do
    (import lispyn.data_profiles)
    (lispyn.data_profiles.profile_array ~values ~@options)))

(defmacro table-profile [frame #* options]
  "Display pandas missingness, duplicate rows, dtypes and numeric infinities."
  `(do
    (import lispyn.data_profiles)
    (lispyn.data_profiles.profile_table ~frame ~@options)))

; Non-mutating pre-training checks for numeric arrays and shaped inputs.

(defmacro assert-finite [values #* options]
  "Reject NaN and infinity in NumPy numeric arrays and PyTorch tensors."
  `(do
    (import lispyn.validation)
    (lispyn.validation.assert_finite ~values ~@options)))

(defmacro assert-shape [values expected #* options]
  "Check NumPy, pandas or PyTorch dimensions before an experiment."
  `(do
    (import lispyn.validation)
    (lispyn.validation.assert_shape ~values ~expected ~@options)))

; Evaluation integrity: prevent partition leakage and reject malformed scores.

(defmacro assert-disjoint [left right #* options]
  "Reject overlapping sample IDs before using evaluation metrics."
  `(do
    (import lispyn.evaluation_guards)
    (lispyn.evaluation_guards.assert_disjoint ~left ~right ~@options)))

(defmacro assert-probabilities [values #* options]
  "Validate finite, bounded binary or normalized multiclass probabilities."
  `(do
    (import lispyn.evaluation_guards)
    (lispyn.evaluation_guards.assert_probabilities ~values ~@options)))

; Quickly inspect AI classification dataset balance before training.

(defmacro label-balance [labels #* options]
  "Display aggregate per-class counts and shares in Jupyter or terminal Hy."
  `(do
    (import lispyn.label_balance)
    (lispyn.label_balance.label_balance ~labels ~@options)))

(defmacro plot-label-balance [labels #* options]
  "Plot per-class sample counts with Matplotlib, without a sklearn dependency."
  `(do
    (import lispyn.label_balance)
    (lispyn.label_balance.plot_label_balance ~labels ~@options)))

; Rebalance classification losses without importing PyTorch just to count labels.

(defmacro class-weights [labels #* options]
  "Return balanced inverse-frequency weights in explicit model class-index order."
  `(do
    (import lispyn.label_balance)
    (lispyn.label_balance.balanced_class_weights ~labels ~@options)))

; Tune binary probability cutoffs using validation data, not training or test labels.

(defmacro threshold-report [actual scores #* options]
  "Display aggregate precision, recall, F1 and balanced accuracy by cutoff."
  `(do
    (import lispyn.thresholds)
    (lispyn.thresholds.threshold_report ~actual ~scores ~@options)))

(defmacro plot-threshold-metrics [actual scores #* options]
  "Plot binary threshold trade-offs and the best selected validation cutoff."
  `(do
    (import lispyn.thresholds)
    (lispyn.thresholds.plot_threshold_metrics ~actual ~scores ~@options)))

; Optional, native PyTorch controls for custom Hy training loops.

(defmacro with-no-grad [#* body]
  "Run a Hy body with autograd disabled, restoring the prior grad state."
  `(do
    (import torch)
    (with [(torch.no_grad)]
      ~@body)))

(defmacro with-autocast [device-type dtype #* body]
  "Autocast a forward-pass body using an explicit PyTorch device and dtype."
  `(do
    (import torch)
    (with [(torch.autocast ~device-type :dtype ~dtype)]
      ~@body)))

(defmacro clip-gradients [model max-norm #* options]
  "Clip already computed model gradients; return their original global norm."
  `(do
    (import lispyn.torch_utils)
    (lispyn.torch_utils.clip_gradients ~model ~max-norm ~@options)))

(defmacro freeze-module [module]
  "Disable gradients for a PyTorch module without changing its training mode."
  `(do
    (import lispyn.torch_utils)
    (lispyn.torch_utils.set-module-trainable ~module False)))

(defmacro unfreeze-module [module]
  "Re-enable gradients for a PyTorch module without changing its training mode."
  `(do
    (import lispyn.torch_utils)
    (lispyn.torch_utils.set-module-trainable ~module True)))

(defmacro parameter-count [module]
  "Return total, trainable and frozen parameter element counts."
  `(do
    (import lispyn.torch_utils)
    (lispyn.torch_utils.parameter-counts ~module)))

(defmacro move-to-device [batch device #* options]
  "Move tensors in nested batch collections to a PyTorch device."
  `(do
    (import lispyn.torch_utils)
    (lispyn.torch_utils.move-to-device ~batch ~device ~@options)))

(defmacro save-model [model path #* options]
  "Save native PyTorch model/optimizer state dictionaries and optional history."
  `(do
    (import lispyn.training)
    (lispyn.training.save-checkpoint ~model ~path ~@options)))

(defmacro load-model [model path #* options]
  "Restore trusted state dictionaries with torch.load(weights_only=True)."
  `(do
    (import lispyn.training)
    (lispyn.training.load-checkpoint ~model ~path ~@options)))

; Additional opt-in PyTorch diagnostics and safe, scoped execution.

(defmacro gradient-report [model]
  "Summarize dense gradients without changing or exporting them."
  `(do
    (import lispyn.torch_diagnostics)
    (lispyn.torch_diagnostics.gradient-report ~model)))

(defmacro assert-gradients [model #* options]
  "Reject nonfinite gradients or unexpected absence of backward gradients."
  `(do
    (import lispyn.torch_diagnostics)
    (lispyn.torch_diagnostics.assert-gradients ~model ~@options)))

(defmacro model-footprint [model]
  "Estimate nominal parameter/buffer storage, excluding activation memory."
  `(do
    (import lispyn.torch_diagnostics)
    (lispyn.torch_diagnostics.model-footprint ~model)))

(defmacro predict-batches [model batches #* options]
  "Collect CPU logits from a loader in inference mode, restoring train/eval."
  `(do
    (import lispyn.torch_diagnostics)
    (lispyn.torch_diagnostics.predict-batches ~model ~batches ~@options)))

(defmacro with-parameter-freeze [module #* body]
  "Expand a scoped parameter freeze with hygienic names and try/finally."
  (setv model-symbol (hy.gensym "lispyn_module"))
  (setv states-symbol (hy.gensym "lispyn_flags"))
  (setv param-symbol (hy.gensym "lispyn_param"))
  (setv flag-symbol (hy.gensym "lispyn_flag"))
  `(do
    (setv ~model-symbol ~module)
    (setv ~states-symbol
      (list (map (fn [~param-symbol]
                   (tuple [~param-symbol (getattr ~param-symbol "requires_grad")]))
                 (.parameters ~model-symbol))))
    (try
      (for [[~param-symbol ~flag-symbol] ~states-symbol]
        (.requires_grad_ ~param-symbol False))
      ~@body
      (finally
        (for [[~param-symbol ~flag-symbol] ~states-symbol]
          (.requires_grad_ ~param-symbol ~flag-symbol))))))

(defmacro with-determinism [#* body]
  "Expand a deterministic PyTorch scope; restore both policy flags."
  (setv enabled-symbol (hy.gensym "lispyn_deterministic"))
  (setv warn-symbol (hy.gensym "lispyn_warn_only"))
  `(do
    (import torch)
    (setv ~enabled-symbol (torch.are_deterministic_algorithms_enabled))
    (setv ~warn-symbol (torch.is_deterministic_algorithms_warn_only_enabled))
    (try
      (torch.use_deterministic_algorithms True)
      ~@body
      (finally
        (torch.use_deterministic_algorithms
          ~enabled-symbol :warn-only ~warn-symbol)))))

(defmacro with-anomaly-detection [#* body]
  "Trace the backward operation producing NaNs; slower debugging only."
  `(do
    (import torch)
    (with [(torch.autograd.detect_anomaly)]
      ~@body)))

; Native, inspectable PyTorch factories and Matplotlib plot builders.

(defmacro defmodel [name parameters #* layers]
  "Define a callable that creates a fresh native torch.nn.Sequential model."
  (if (or (not (isinstance parameters hy.models.List))
          (= (len layers) 0))
    (raise (ValueError "defmodel expects [parameters] and at least one layer"))
    None)
  `(defn ~name ~parameters
    (import torch)
    (torch.nn.Sequential ~@layers)))

(defmacro defoptimizer [name parameters constructor #* options]
  "Define a native optimizer factory receiving [model learning-rate]."
  (if (or (not (isinstance parameters hy.models.List))
          (!= (len parameters) 2)
          (not (all (map (fn [item] (isinstance item hy.models.Symbol)) parameters))))
    (raise (ValueError "defoptimizer expects [model learning-rate] symbols"))
    None)
  (setv model-symbol (get parameters 0))
  (setv lr-symbol (get parameters 1))
  `(defn ~name ~parameters
    (~constructor (.parameters ~model-symbol) :lr ~lr-symbol ~@options)))

(defmacro with-figure [bindings #* body]
  "Create a scoped Matplotlib (figure, axes), returning the figure."
  (if (or (not (isinstance bindings hy.models.List))
          (!= (len bindings) 2)
          (not (all (map (fn [item] (isinstance item hy.models.Symbol)) bindings)))
          (= (len body) 0))
    (raise (ValueError "with-figure expects [figure axes] and a nonempty body"))
    None)
  (setv figure-symbol (get bindings 0))
  (setv axes-symbol (get bindings 1))
  (setv pyplot-symbol (hy.gensym "lispyn_pyplot"))
  `(do
    (import matplotlib.pyplot :as ~pyplot-symbol)
    (setv [~figure-symbol ~axes-symbol] (.subplots ~pyplot-symbol))
    (try
      ~@body
      (.tight_layout ~figure-symbol)
      ~figure-symbol
      (except [BaseException]
        (.close ~pyplot-symbol ~figure-symbol)
        (raise)))))

(defmacro defplot [name parameters bindings #* body]
  "Define a reusable Matplotlib plot function, closing figures on failure."
  (if (or (not (isinstance parameters hy.models.List))
          (not (isinstance bindings hy.models.List))
          (!= (len bindings) 2)
          (not (all (map (fn [item] (isinstance item hy.models.Symbol)) bindings)))
          (= (len body) 0))
    (raise (ValueError "defplot expects [parameters] [figure axes] and body"))
    None)
  (setv figure-symbol (get bindings 0))
  (setv axes-symbol (get bindings 1))
  (setv pyplot-symbol (hy.gensym "lispyn_pyplot"))
  `(defn ~name ~parameters
    (import matplotlib.pyplot :as ~pyplot-symbol)
    (setv [~figure-symbol ~axes-symbol] (.subplots ~pyplot-symbol))
    (try
      ~@body
      (.tight_layout ~figure-symbol)
      ~figure-symbol
      (except [BaseException]
        (.close ~pyplot-symbol ~figure-symbol)
        (raise)))))

; Macro-first experiment composition: native PyTorch code, no runtime backend.

(defmacro with-training [model #* body]
  "Temporarily enter train mode with autograd; restore every module mode."
  (setv model-symbol (hy.gensym "lispyn_model"))
  (setv states-symbol (hy.gensym "lispyn_states"))
  (setv child-symbol (hy.gensym "lispyn_child"))
  (setv flag-symbol (hy.gensym "lispyn_flag"))
  `(do
    (import torch)
    (setv ~model-symbol ~model)
    (setv ~states-symbol
      (list (map (fn [~child-symbol]
                   (tuple [~child-symbol (getattr ~child-symbol "training")]))
                 (.modules ~model-symbol))))
    (try
      (.train ~model-symbol)
      (with [(torch.enable_grad)]
        ~@body)
      (finally
        (for [[~child-symbol ~flag-symbol] ~states-symbol]
          (setattr ~child-symbol "training" ~flag-symbol))))))

(defmacro with-grad [#* body]
  "Re-enable autograd only inside BODY and restore the outer grad state."
  `(do
    (import torch)
    (with [(torch.enable_grad)]
      ~@body)))

(defmacro with-rng-fork [#* body]
  "Fork the CPU PyTorch RNG state within BODY, restoring it afterwards."
  `(do
    (import torch)
    (with [(torch.random.fork_rng :devices [])]
      ~@body)))

(defmacro optimizer-step [optimizer loss]
  "Zero gradients, backward once, step once; return a detached scalar loss."
  (setv optimizer-symbol (hy.gensym "lispyn_optimizer"))
  (setv loss-symbol (hy.gensym "lispyn_loss"))
  `(do
    (setv ~optimizer-symbol ~optimizer)
    (.zero_grad ~optimizer-symbol)
    (setv ~loss-symbol ~loss)
    (.backward ~loss-symbol)
    (.step ~optimizer-symbol)
    (.detach ~loss-symbol)))

(defmacro deftrain-step [name model optimizer criterion]
  "Generate a native Python training-step function taking inputs and labels."
  (setv inputs-symbol (hy.gensym "lispyn_inputs"))
  (setv labels-symbol (hy.gensym "lispyn_labels"))
  (setv model-symbol (hy.gensym "lispyn_model"))
  (setv optimizer-symbol (hy.gensym "lispyn_optimizer"))
  (setv criterion-symbol (hy.gensym "lispyn_criterion"))
  (setv loss-symbol (hy.gensym "lispyn_loss"))
  `(defn ~name [~inputs-symbol ~labels-symbol]
    (setv ~model-symbol ~model)
    (setv ~optimizer-symbol ~optimizer)
    (setv ~criterion-symbol ~criterion)
    (.zero_grad ~optimizer-symbol)
    (setv ~loss-symbol
      (~criterion-symbol (~model-symbol ~inputs-symbol) ~labels-symbol))
    (.backward ~loss-symbol)
    (.step ~optimizer-symbol)
    (float (.item (.detach ~loss-symbol)))))

(defmacro collect-epochs [epoch count #* body]
  "Run BODY for range(COUNT), collecting its last expression for each epoch."
  (setv results-symbol (hy.gensym "lispyn_results"))
  `(do
    (setv ~results-symbol [])
    (for [~epoch (range ~count)]
      (.append ~results-symbol (do ~@body)))
    ~results-symbol))
