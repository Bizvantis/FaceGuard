from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from faceguard import models
from faceguard.gui import icons, theme
from faceguard.gui.charts import SCATTER_COLOR_LIMIT, BarChart, HistogramChart, ScatterChart
from faceguard.gui.context import AppContext
from faceguard.gui.pages.base import Page, header, page_layout, scrolling
from faceguard.gui.widgets import StatTile, button, card, label, run_job
from faceguard.recognition.analysis import AnalysisReport, analyze

METHOD_NOTES = {
    "Top-3 cosine (FaceGuard)": "Mean of the 3 closest enrolled samples per person - what FaceGuard uses live.",
    "1-NN cosine": "Single most similar sample.",
    "Nearest centroid": "Similarity to each person's average faceprint.",
    "Linear SVM": "Support-vector machine trained on the embeddings (scikit-learn, cross-validated).",
    "k-NN (k=5, cosine)": "Majority vote of the 5 nearest samples (scikit-learn, cross-validated).",
    "Logistic regression": "Linear classifier trained on the embeddings (scikit-learn, cross-validated).",
}

PIPELINE = (
    "1. YuNet CNN finds faces and 5 landmarks (eyes, nose, mouth corners) in every frame.\n"
    "2. A similarity transform aligns each face to a canonical 112x112 pose.\n"
    "3. SFace (a MobileFaceNet-style CNN trained with a margin-based loss) maps it to a 128-number "
    "faceprint; same person -> vectors point the same way.\n"
    "4. Open-set matching: cosine similarity to every enrolled sample, top-3 mean per person, smoothed over "
    "time per tracked face; below the threshold means 'unknown'.\n"
    "5. MediaPipe's face mesh (478 3-D points, 52 blendshapes) powers the blink/smile liveness checks; "
    "head turns are measured geometrically, which a flat photo cannot fake.\n"
    "6. Adaptive learning adds very confident live samples over time, capped so the original enrollment always dominates."
)


class LabPage(Page):
    key = "lab"
    title = "Model Lab"
    glyph = icons.CHART

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        self.report: AnalysisReport | None = None
        self._stale = True
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        content.setObjectName("Page")
        lay = page_layout(content)
        controls = QHBoxLayout()
        self.proj = QComboBox()
        self.proj.addItem("PCA map (instant)", "pca")
        self.proj.addItem("t-SNE map (nonlinear)", "tsne")
        self.proj.currentIndexChanged.connect(self.run_analysis)
        self.run_btn = button("Re-analyse", "primary", icons.REFRESH)
        self.run_btn.clicked.connect(self.run_analysis)
        controls.addWidget(self.proj)
        controls.addWidget(self.run_btn)
        cw = QWidget()
        cw.setLayout(controls)
        lay.addWidget(header("Model Lab", "How well the neural network separates the people you've enrolled - "
                             "measured on your own data.", cw))

        tiles = QGridLayout()
        tiles.setSpacing(12)
        self.t_people = StatTile("People enrolled", "-", "")
        self.t_acc = StatTile("Identification accuracy", "-", "leave-one-out, FaceGuard matcher")
        self.t_eer = StatTile("Equal error rate", "-", "false accepts = false rejects")
        self.t_thr = StatTile("Recommended threshold", "-", "")
        for i, t in enumerate((self.t_people, self.t_acc, self.t_eer, self.t_thr)):
            tiles.addWidget(t, 0, i)
        lay.addLayout(tiles)
        apply_row = QHBoxLayout()
        self.apply_btn = button("Use recommended threshold", None, icons.CHECK)
        self.apply_btn.clicked.connect(self._apply_threshold)
        apply_row.addStretch(1)
        apply_row.addWidget(self.apply_btn)
        lay.addLayout(apply_row)

        charts = QGridLayout()
        charts.setSpacing(16)
        map_card, ml = card("Embedding space", "Every enrolled face sample projected from 128 dimensions to 2.")
        self.highlight = QComboBox()
        self.highlight.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.highlight.setMinimumWidth(240)
        self.highlight.currentIndexChanged.connect(self._redraw_map)
        ml.addWidget(self.highlight, 0, Qt.AlignmentFlag.AlignLeft)
        self.scatter = ScatterChart()
        ml.addWidget(self.scatter, 1)
        charts.addWidget(map_card, 0, 0)
        hist_card, hl = card("Similarity scores", "Pairs of the same person should sit right of the threshold, "
                             "pairs of different people to the left.")
        self.hist = HistogramChart()
        hl.addWidget(self.hist, 1)
        charts.addWidget(hist_card, 0, 1)
        charts.setColumnStretch(0, 1)
        charts.setColumnStretch(1, 1)
        lay.addLayout(charts)

        cls_card, cl = card("Classifier comparison", "Accuracy of different machine-learning classifiers on top of the "
                            "same deep embeddings (needs 2+ people).")
        self.bars = BarChart()
        self.bars.empty_text = "Enroll at least two people to compare classifiers"
        cl.addWidget(self.bars)
        lay.addWidget(cls_card)

        table_card, tl = card("Per person", "Consistency = how alike a person's own samples are (higher is better).")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Person", "Samples", "Consistency", "Worst sample", "Most similar other person"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(120)
        tl.addWidget(self.table)
        self.notes = label("", "secondary", wrap=True)
        tl.addWidget(self.notes)
        lay.addWidget(table_card)

        model_card, mcl = card("Under the hood")
        for spec in models.MODELS.values():
            mcl.addWidget(label(f"{spec.title}  -  {spec.source}, {spec.license} licence, {spec.size / 1e6:.1f} MB", "secondary", wrap=True))
        mcl.addWidget(label(PIPELINE, "muted", wrap=True))
        lay.addWidget(model_card)
        lay.addStretch(1)
        outer.addWidget(scrolling(content))
        ctx.db_changed.connect(self._mark_stale)
        ctx.settings_changed.connect(self._settings_changed)

    def _mark_stale(self) -> None:
        self._stale = True
        if self.isVisible():
            self.run_analysis()

    def _settings_changed(self) -> None:
        if self.report is not None:
            self._show(self.report)

    def on_show(self) -> None:
        if self._stale:
            self.run_analysis()

    def run_analysis(self) -> None:
        self._stale = False
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Analysing...")
        s = self.ctx.settings
        persons = self.ctx.db.persons_snapshot()
        run_job(analyze, persons, s.match_threshold, s.top_k, self.proj.currentData(),
                on_done=self._show, on_error=self._failed)

    def _failed(self, error: str) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Re-analyse")
        self.ctx.toast.emit("error", f"Analysis failed: {error}")

    def _show(self, r: AnalysisReport) -> None:
        self.report = r
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Re-analyse")
        s = self.ctx.settings
        self.t_people.set(str(len(r.persons)), f"{len(r.labels)} face samples")
        main = next((v for k, v in r.methods.items() if k.startswith("Top-")), None)
        self.t_acc.set(f"{main * 100:.1f}%" if main is not None else "-",
                       "leave-one-out, FaceGuard matcher" if main is not None else "needs 2+ people")
        self.t_eer.set(f"{r.eer * 100:.2f}%" if r.eer is not None else "-",
                       f"at similarity {r.eer_threshold:.2f}  |  ROC AUC {r.auc:.4f}" if r.eer is not None else "needs 2+ people")
        self.t_thr.set(f"{r.recommended_threshold:.2f}", f"currently {s.match_threshold:.2f} ({s.security_level})")
        self.apply_btn.setEnabled(len(r.persons) >= 2 and abs(r.recommended_threshold - s.match_threshold) > 0.005)

        self.highlight.blockSignals(True)
        self.highlight.clear()
        many = len(r.names) > SCATTER_COLOR_LIMIT
        self.highlight.setVisible(many)
        if many:
            for n in r.names:
                self.highlight.addItem(f"Highlight: {n}")
            owner = self.ctx.db.owner()
            self.highlight.setCurrentIndex(r.names.index(owner.name) if owner and owner.name in r.names else 0)
        self.highlight.blockSignals(False)
        self._redraw_map()

        series = [("Same person", r.genuine, theme.SERIES[0]), ("Different people", r.impostor, theme.SERIES[1])]
        self.hist.set_data(series, s.match_threshold)
        self.bars.set_data([(k, v, METHOD_NOTES.get(k, "")) for k, v in r.methods.items()])

        self.table.setRowCount(len(r.persons))
        for i, p in enumerate(r.persons):
            cells = [p.name, f"{p.samples}" + (f" ({p.adaptive} learned)" if p.adaptive else ""), f"{p.consistency:.3f}",
                     f"{p.worst:.3f}", f"{p.nearest_other} ({p.nearest_other_sim:.2f})" if p.nearest_other else "-"]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                self.table.setItem(i, j, item)
        self.table.setFixedHeight(44 + 34 * max(1, len(r.persons)))
        self.notes.setText("\n".join(r.notes) if r.notes else f"Analysed in {r.elapsed:.2f} s using the {r.projection_method} projection.")

    def _redraw_map(self) -> None:
        r = self.report
        if r is None:
            return
        hi = self.highlight.currentIndex() if len(r.names) > SCATTER_COLOR_LIMIT else None
        self.scatter.set_data(r.projection, r.labels, r.names, hi)

    def _apply_threshold(self) -> None:
        if self.report is None:
            return
        s = self.ctx.settings
        s.security_level = "custom"
        s.custom_threshold = float(self.report.recommended_threshold)
        self.ctx.save_settings()
        self.ctx.toast.emit("good", f"Threshold set to {s.custom_threshold:.2f}")
