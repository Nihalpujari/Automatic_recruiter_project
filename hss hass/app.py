import os
import pymysql

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from models import db, User, CandidateInfo
from analysis_module import get_top_candidates  # ✅ your pipeline output

# Fix for MySQLdb import issues when using PyMySQL
pymysql.install_as_MySQLdb()

app = Flask(__name__)

# =========================
# CONFIG (use env vars; safer than hardcoding)
# =========================
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "secret123")

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "1803")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "recruitment_db")

app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =========================
# ROUTES
# =========================

@app.route("/")
@login_required
def home():
    if current_user.role == "coo":
        return redirect(url_for("coo_dashboard"))
    if current_user.role == "manager":
        return redirect(url_for("manager_dashboard"))
    return redirect(url_for("candidate_dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            if user.role == "manager" and not user.is_approved:
                flash("Your account is still pending COO approval.")
                return redirect(url_for("login"))

            login_user(user)
            return redirect(url_for("home"))

        flash("Invalid credentials.")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Error: Passwords do not match!")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password, method="scrypt")
        new_user = User(
            username=username,
            email=email,
            password=hashed_pw,
            role="manager",
            is_approved=False
        )

        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Account created! Access is pending COO approval.")
            return redirect(url_for("login"))
        except Exception:
            db.session.rollback()
            flash("Registration failed. Username or Email already exists.")
    return render_template("register.html")


@app.route("/coo", methods=["GET", "POST"])
@login_required
def coo_dashboard():
    if current_user.role != "coo":
        return "Access Denied", 403

    if request.method == "POST":
        mgr_id = request.form.get("approve_manager_id")
        if mgr_id:
            m = User.query.get(mgr_id)
            if m and m.role == "manager":
                m.is_approved = True
                db.session.commit()
                flash(f"Approved access for {m.username}")

    pending = User.query.filter_by(role="manager", is_approved=False).all()

    hires = (
        db.session.query(CandidateInfo, User.username)
        .join(User, CandidateInfo.manager_id == User.id)
        .filter(CandidateInfo.status == "hired")
        .all()
    )
    rejections = (
        db.session.query(CandidateInfo, User.username)
        .join(User, CandidateInfo.manager_id == User.id)
        .filter(CandidateInfo.status == "rejected")
        .all()
    )

    return render_template("coo_dashboard.html", pending=pending, hires=hires, rejections=rejections)


@app.route("/manager", methods=["GET", "POST"])
@login_required
def manager_dashboard():
    if current_user.role != "manager":
        return "Access Denied", 403

    candidate_data = get_top_candidates()

    existing_decisions = CandidateInfo.query.all()
    decisions = {(c.candidate_name, c.platform): c.status for c in existing_decisions}

    if request.method == "POST":
        btn_action = request.form.get("btn_action")
        name = request.form.get("name")
        role = request.form.get("role")          # ✅ job_role from your pipeline/UI
        platform = request.form.get("platform")
        reason = request.form.get("reason")

        if not name or not platform:
            flash("Invalid request: missing candidate/platform.")
            return redirect(url_for("manager_dashboard"))

        if (name, platform) in decisions:
            flash(f"{name} has already been processed.")
            return redirect(url_for("manager_dashboard"))

        try:
            if btn_action == "hire":
                existing_user = User.query.filter_by(username=name).first()

                if not existing_user:
                    new_user = User(
                        username=name,
                        email=f"{name.replace(' ', '').lower()}@hire.com",
                        password=generate_password_hash("pass123"),
                        role="candidate",
                        is_approved=True
                    )
                    db.session.add(new_user)
                    db.session.flush()
                    target_user_id = new_user.id
                else:
                    target_user_id = existing_user.id

                info = CandidateInfo(
                    user_id=target_user_id,
                    manager_id=current_user.id,
                    candidate_name=name,
                    platform=platform,
                    role_hired_for=role,
                    reason=reason,
                    status="hired"
                )
                flash(f"Successfully hired {name}!")

            else:
                info = CandidateInfo(
                    user_id=None,
                    manager_id=current_user.id,
                    candidate_name=name,
                    platform=platform,
                    role_hired_for=role,
                    reason=reason,
                    status="rejected"
                )
                flash(f"Candidate {name} rejected.")

            db.session.add(info)
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            print(f"Database Error: {e}")
            flash("An error occurred while processing the candidate. They might already exist.")

        return redirect(url_for("manager_dashboard"))

    return render_template("manager_dashboard.html", data=candidate_data, decisions=decisions)


@app.route("/candidate")
@login_required
def candidate_dashboard():
    if current_user.role != "candidate":
        return "Access Denied", 403

    info = CandidateInfo.query.filter_by(user_id=current_user.id).first()
    manager = User.query.get(info.manager_id) if info else None
    return render_template("candidate_dashboard.html", info=info, manager=manager)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# =========================
# INITIAL SETUP
# =========================
with app.app_context():
    db.create_all()

    if not User.query.filter_by(username="coo").first():
        coo = User(
            username="coo",
            email="coo@admin.com",
            password=generate_password_hash("admin123", method="scrypt"),
            role="coo",
            is_approved=True
        )
        db.session.add(coo)
        db.session.commit()


if __name__ == "__main__":
    app.run(debug=True)