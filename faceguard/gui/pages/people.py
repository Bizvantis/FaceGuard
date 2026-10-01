from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QVBoxLayout, QWidget

from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext
from faceguard.gui.pages.base import Page, header, page_layout, scrolling
from faceguard.gui.widgets import Pill, button, card, label, rounded_pixmap
from faceguard.recognition.database import SOURCE_ADAPTIVE, Person, decode_thumbnail


class PersonCard(QFrame):
    def __init__(self, page: "PeoplePage", person: Person) -> None:
        super().__init__()
        self.setObjectName("Card")
        self.page, self.person = page, person
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(16)
        thumbs = [decode_thumbnail(t) for t in person.thumbnails[:1]]
        pic = QLabel()
        pic.setPixmap(rounded_pixmap(thumbs[0] if thumbs else None, 72, 16))
        lay.addWidget(pic, 0, Qt.AlignmentFlag.AlignTop)

        col = QVBoxLayout()
        col.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label(person.name, "h2"))
        if person.is_owner:
            top.addWidget(Pill("Owner", theme.ACCENT))
        elif person.trusted:
            top.addWidget(Pill("Trusted", theme.GOOD))
        else:
            top.addWidget(Pill("Known - not trusted", theme.SERIOUS))
        top.addStretch(1)
        col.addLayout(top)
        adaptive = person.count(SOURCE_ADAPTIVE)
        enrolled = time.strftime("%d %b %Y", time.localtime(person.created))
        detail = f"{person.count()} face samples"
        if adaptive:
            detail += f" ({adaptive} learned automatically)"
        detail += f"  |  enrolled {enrolled}"
        col.addWidget(label(detail, "muted"))
        role = ("Owner: the guard protects this person's PC." if person.is_owner else
                "Can use this PC without triggering the guard." if person.trusted else
                "Recognised, but treated like a stranger by the guard.")
        col.addWidget(label(role, "small"))

        actions = QHBoxLayout()
        actions.setSpacing(6)
        add = button("Add samples", None, icons.ADD, "Capture more samples to improve recognition")
        add.clicked.connect(lambda: page.ctx.enroll_person.emit(person.name))
        actions.addWidget(add)
        if not person.is_owner:
            own = button("Make owner", "ghost", icons.STAR)
            own.clicked.connect(lambda: page.set_owner(person))
            actions.addWidget(own)
            trust = button("Untrust" if person.trusted else "Trust", "ghost", icons.SHIELD)
            trust.clicked.connect(lambda: page.toggle_trust(person))
            actions.addWidget(trust)
        rename = button("Rename", "ghost", icons.EDIT)
        rename.clicked.connect(lambda: page.rename(person))
        actions.addWidget(rename)
        if adaptive:
            reset = button("Reset learning", "ghost", icons.REFRESH, "Remove automatically learned samples")
            reset.clicked.connect(lambda: page.reset_adaptive(person))
            actions.addWidget(reset)
        actions.addStretch(1)
        delete = button("Delete", "danger", icons.DELETE)
        delete.clicked.connect(lambda: page.delete(person))
        actions.addWidget(delete)
        col.addLayout(actions)
        lay.addLayout(col, 1)


class PeoplePage(Page):
    key = "people"
    title = "People"
    glyph = icons.PEOPLE

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        content.setObjectName("Page")
        self.lay = page_layout(content)
        add = button("Enroll someone", "primary", icons.ADD_PERSON)
        add.clicked.connect(lambda: ctx.enroll_person.emit(""))
        self.lay.addWidget(header("People", "Everyone FaceGuard can recognise. Face data is encrypted with your Windows "
                                  "account (DPAPI) and never leaves this PC.", add))
        self.list = QVBoxLayout()
        self.list.setSpacing(12)
        self.lay.addLayout(self.list)
        self.lay.addStretch(1)
        outer.addWidget(scrolling(content))
        ctx.db_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        while self.list.count():
            item = self.list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        persons = sorted(self.ctx.db.persons, key=lambda p: (not p.is_owner, not p.trusted, p.name.casefold()))
        if not persons:
            c, cl = card("No one enrolled yet", "Start by enrolling yourself - you'll become the owner that the guard protects.")
            go = button("Enroll my face", "primary", icons.ADD_PERSON)
            go.clicked.connect(lambda: self.ctx.enroll_person.emit(""))
            cl.addWidget(go, 0, Qt.AlignmentFlag.AlignLeft)
            self.list.addWidget(c)
            return
        for p in persons:
            self.list.addWidget(PersonCard(self, p))

    # ------------------------------------------------------------ actions
    def set_owner(self, person: Person) -> None:
        self.ctx.db.set_owner(person.id)
        self.ctx.toast.emit("good", f"{person.name} is now the owner")

    def toggle_trust(self, person: Person) -> None:
        self.ctx.db.set_trusted(person.id, not person.trusted)

    def rename(self, person: Person) -> None:
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=person.name)
        if ok and name.strip():
            try:
                self.ctx.db.rename(person.id, name)
            except ValueError as exc:
                QMessageBox.warning(self, "Rename", str(exc))

    def reset_adaptive(self, person: Person) -> None:
        n = self.ctx.db.clear_adaptive(person.id)
        self.ctx.toast.emit("info", f"Removed {n} automatically learned samples for {person.name}")

    def delete(self, person: Person) -> None:
        answer = QMessageBox.question(
            self, "Delete person", f"Delete {person.name} and all of their face data? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.db.delete(person.id)
            self.ctx.toast.emit("info", f"Deleted {person.name}")
