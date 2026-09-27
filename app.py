import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionAI • CIFAR-10",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CIFAR-10 CLASSES
# ============================================================

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

icons = [
    "✈️",
    "🚗",
    "🐦",
    "🐱",
    "🦌",
    "🐶",
    "🐸",
    "🐴",
    "🚢",
    "🚚"
]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.keras")


model = load_model()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap'
);

/* ==========================================================
   REMOVE STREAMLIT TOP WHITE SPACE
   ========================================================== */

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

.main .block-container {
    padding-top: 0rem !important;
}


/* ==========================================================
   GLOBAL
   ========================================================== */

* {
    font-family: 'Inter', sans-serif;
}

.stApp {

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,0.18),
            transparent 25%
        ),

        radial-gradient(
            circle at 90% 20%,
            rgba(6,182,212,0.15),
            transparent 25%
        ),

        radial-gradient(
            circle at 50% 90%,
            rgba(168,85,247,0.12),
            transparent 30%
        ),

        #050816;

    color: #f8fafc;
}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.block-container {

    max-width: 1250px;

    padding-left: 35px !important;
    padding-right: 35px !important;

    padding-top: 0px !important;
    padding-bottom: 40px !important;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #080d1f,
            #050816
        );

    border-right:
        1px solid rgba(129,140,248,0.15);
}

[data-testid="stSidebar"] * {

    color: #cbd5e1;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {

    text-align: center;

    padding-top: 10px;
    padding-bottom: 30px;
}


.status-pill {

    display: inline-block;

    padding:
        7px 18px;

    border-radius:
        50px;

    background:
        rgba(16,185,129,0.08);

    border:
        1px solid
        rgba(52,211,153,0.35);

    color:
        #34d399;

    font-size:
        12px;

    font-weight:
        700;

    letter-spacing:
        1px;

    box-shadow:
        0 0 20px
        rgba(52,211,153,0.15);
}


.hero-title {

    font-size:
        65px;

    font-weight:
        900;

    margin:
        18px 0 10px;

    background:
        linear-gradient(
            90deg,
            #818cf8,
            #22d3ee,
            #c084fc,
            #818cf8
        );

    background-size:
        300% 300%;

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;

    animation:
        gradientFlow 7s ease infinite;
}


@keyframes gradientFlow {

    0% {
        background-position:
            0% 50%;
    }

    50% {
        background-position:
            100% 50%;
    }

    100% {
        background-position:
            0% 50%;
    }

}


.hero-subtitle {

    color:
        #94a3b8;

    font-size:
        17px;
}


/* ==========================================================
   PARTICLES
   ========================================================== */

.stApp::before,
.stApp::after {

    content:
        "";

    position:
        fixed;

    width:
        5px;

    height:
        5px;

    border-radius:
        50%;

    background:
        #818cf8;

    box-shadow:

        100px 100px #22d3ee,
        250px 200px #a78bfa,
        400px 80px #818cf8,
        550px 300px #22d3ee,
        700px 150px #a78bfa,
        850px 350px #818cf8,
        1000px 100px #22d3ee,
        1150px 500px #818cf8,
        300px 500px #22d3ee,
        650px 600px #a78bfa;

    opacity:
        0.35;

    animation:
        floatingParticles
        12s linear infinite;

    pointer-events:
        none;

    z-index:
        0;
}


@keyframes floatingParticles {

    0% {
        transform:
            translateY(0);
    }

    50% {
        transform:
            translateY(-40px);
    }

    100% {
        transform:
            translateY(0);
    }

}


/* ==========================================================
   STAT CARDS
   ========================================================== */

.stat-card {

    text-align:
        center;

    background:
        rgba(15,23,42,0.65);

    border:
        1px solid
        rgba(148,163,184,0.12);

    border-radius:
        18px;

    padding:
        20px;

    transition:
        0.3s;
}


.stat-card:hover {

    transform:
        translateY(-5px);

    border-color:
        rgba(129,140,248,0.4);

    box-shadow:
        0 10px 30px
        rgba(99,102,241,0.12);
}


.stat-number {

    font-size:
        25px;

    font-weight:
        900;

    color:
        #f8fafc;
}


.stat-label {

    color:
        #64748b;

    font-size:
        10px;

    letter-spacing:
        1px;

    margin-top:
        5px;
}


/* ==========================================================
   SECTION
   ========================================================== */

.section-title {

    font-size:
        28px;

    font-weight:
        800;

    margin-top:
        30px;

    margin-bottom:
        5px;
}


.section-subtitle {

    color:
        #64748b;

    margin-bottom:
        20px;
}


/* ==========================================================
   GLASS CARD
   ========================================================== */

.glass-card {

    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.72),
            rgba(15,23,42,0.55)
        );

    border:
        1px solid
        rgba(148,163,184,0.14);

    border-radius:
        24px;

    padding:
        25px;

    box-shadow:
        0 20px 60px
        rgba(0,0,0,0.25);

    backdrop-filter:
        blur(15px);

    margin-bottom:
        20px;
}


/* ==========================================================
   UPLOAD BOX
   ========================================================== */

[data-testid="stFileUploader"] {

    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.7),
            rgba(15,23,42,0.5)
        );

    border:
        2px dashed
        rgba(129,140,248,0.4);

    border-radius:
        22px;

    padding:
        20px;

    transition:
        0.3s;
}


[data-testid="stFileUploader"]:hover {

    border-color:
        #818cf8;

    box-shadow:
        0 0 35px
        rgba(99,102,241,0.18);
}


/* ==========================================================
   PREDICTION CARD
   ========================================================== */

.prediction-card {

    text-align:
        center;

    background:
        radial-gradient(
            circle at center,
            rgba(99,102,241,0.18),
            rgba(15,23,42,0.75)
        );

    border:
        1px solid
        rgba(129,140,248,0.35);

    border-radius:
        26px;

    padding:
        30px;

    box-shadow:
        0 0 50px
        rgba(99,102,241,0.12);
}


.prediction-label {

    font-size:
        11px;

    color:
        #64748b;

    letter-spacing:
        2px;

    font-weight:
        700;
}


.prediction-name {

    font-size:
        40px;

    font-weight:
        900;

    margin:
        10px 0 20px;

    color:
        #818cf8;
}


/* ==========================================================
   CIRCULAR CONFIDENCE METER
   ========================================================== */

.meter {

    width:
        180px;

    height:
        180px;

    border-radius:
        50%;

    margin:
        15px auto 20px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    position:
        relative;

    background:
        conic-gradient(
            #818cf8
            var(--confidence),

            rgba(51,65,85,0.45)
            var(--confidence)
        );

    box-shadow:
        0 0 40px
        rgba(129,140,248,0.2);
}


.meter::before {

    content:
        "";

    position:
        absolute;

    width:
        140px;

    height:
        140px;

    border-radius:
        50%;

    background:
        #0b1020;
}


.meter-content {

    position:
        relative;

    z-index:
        2;

    text-align:
        center;
}


.meter-number {

    font-size:
        30px;

    font-weight:
        900;

    color:
        #34d399;
}


.meter-label {

    color:
        #64748b;

    font-size:
        10px;

    letter-spacing:
        1px;
}


/* ==========================================================
   TOP PREDICTIONS
   ========================================================== */

.top-card {

    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.65),
            rgba(15,23,42,0.5)
        );

    border:
        1px solid
        rgba(148,163,184,0.12);

    border-radius:
        18px;

    padding:
        17px 20px;

    margin-bottom:
        10px;

    transition:
        0.25s;
}


.top-card:hover {

    transform:
        translateY(-3px);

    border-color:
        rgba(129,140,248,0.4);

    box-shadow:
        0 10px 30px
        rgba(99,102,241,0.12);
}


.rank {

    color:
        #818cf8;

    font-weight:
        900;
}


.class-name {

    color:
        #e2e8f0;

    font-weight:
        700;
}


.probability {

    float:
        right;

    color:
        #22d3ee;

    font-weight:
        800;
}


/* ==========================================================
   INFO CARDS
   ========================================================== */

.info-card {

    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.6),
            rgba(15,23,42,0.5)
        );

    border:
        1px solid
        rgba(148,163,184,0.12);

    border-radius:
        20px;

    padding:
        24px;

    min-height:
        150px;

    transition:
        0.3s;
}


.info-card:hover {

    transform:
        translateY(-5px);

    border-color:
        rgba(129,140,248,0.4);

    box-shadow:
        0 15px 35px
        rgba(99,102,241,0.1);
}


.info-icon {

    font-size:
        30px;
}


.info-title {

    font-size:
        16px;

    font-weight:
        800;

    margin-top:
        10px;
}


.info-text {

    color:
        #64748b;

    font-size:
        13px;

    margin-top:
        8px;

    line-height:
        1.6;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {

    text-align:
        center;

    color:
        #475569;

    padding:
        45px 0 15px;

    font-size:
        12px;
}


/* ==========================================================
   HIDE SOME STREAMLIT UI
   ========================================================== */

#MainMenu {
    visibility:
        hidden;
}

footer {
    visibility:
        hidden;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 768px) {

    .hero-title {
        font-size:
            45px;
    }

    .hero-subtitle {
        font-size:
            14px;
    }

    .block-container {
        padding-left:
            15px !important;

        padding-right:
            15px !important;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div style="
        text-align:center;
        padding:20px 5px 25px;
    ">

        <div style="
            font-size:45px;
            color:#818cf8;
            text-shadow:0 0 20px #6366f1;
        ">
        ◈
        </div>

        <div style="
            font-size:23px;
            font-weight:900;
            color:white;
        ">
        VisionAI
        </div>

        <div style="
            font-size:10px;
            color:#64748b;
            letter-spacing:2px;
        ">
        CIFAR-10 CLASSIFIER
        </div>

    </div>
    """)

    page = st.radio(
        "NAVIGATION",
        [
            "🔮 Image Analyzer",
            "📊 Model Analytics",
            "🧠 About the Project"
        ]
    )

    st.html("""
    <div style="
        background:rgba(30,41,59,0.5);
        border:1px solid rgba(148,163,184,0.12);
        border-radius:15px;
        padding:15px;
        margin-top:20px;
    ">

    <div style="
        color:#818cf8;
        font-weight:700;
        font-size:12px;
    ">
    ● SYSTEM STATUS
    </div>

    <div style="
        color:#64748b;
        font-size:12px;
        margin-top:5px;
    ">
    Neural network online
    </div>

    </div>
    """)

    st.html("""
    <div style="
        background:rgba(30,41,59,0.5);
        border:1px solid rgba(148,163,184,0.12);
        border-radius:15px;
        padding:15px;
        margin-top:12px;
    ">

    <div style="
        color:#818cf8;
        font-weight:700;
        font-size:12px;
    ">
    MODEL
    </div>

    <div style="
        color:#64748b;
        font-size:12px;
        margin-top:5px;
    ">
    CNN • CIFAR-10
    </div>

    </div>
    """)


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="status-pill">
    ● NEURAL NETWORK ONLINE
    </div>

    <div class="hero-title">
    VisionAI
    </div>

    <div class="hero-subtitle">
    Intelligent image recognition powered by
    Convolutional Neural Networks
    </div>

</div>
""")


# ============================================================
# IMAGE ANALYZER
# ============================================================

if page == "🔮 Image Analyzer":

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.html("""
        <div class="stat-card">

            <div class="stat-number">
            10
            </div>

            <div class="stat-label">
            CLASSES
            </div>

        </div>
        """)

    with c2:

        st.html("""
        <div class="stat-card">

            <div class="stat-number">
            32×32
            </div>

            <div class="stat-label">
            INPUT SIZE
            </div>

        </div>
        """)

    with c3:

        st.html("""
        <div class="stat-card">

            <div class="stat-number">
            CNN
            </div>

            <div class="stat-label">
            ARCHITECTURE
            </div>

        </div>
        """)

    with c4:

        st.html("""
        <div class="stat-card">

            <div class="stat-number">
            64.19%
            </div>

            <div class="stat-label">
            TEST ACCURACY
            </div>

        </div>
        """)


    # ========================================================
    # IMAGE ANALYZER TITLE
    # ========================================================

    st.html("""
    <div class="section-title">
    🔮 Image Analyzer
    </div>

    <div class="section-subtitle">
    Upload an image and let the neural network analyze it.
    </div>
    """)


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # Resize

        resized = image.resize(
            (32, 32)
        )


        # Convert

        array = np.array(
            resized
        ).astype(
            "float32"
        ) / 255.0


        # Batch dimension

        array = np.expand_dims(
            array,
            axis=0
        )


        # Predict

        predictions = model.predict(
            array,
            verbose=0
        )[0]


        # Highest prediction

        predicted_index = np.argmax(
            predictions
        )


        predicted_class = class_names[
            predicted_index
        ]


        confidence = (
            predictions[
                predicted_index
            ] * 100
        )


        # Sort

        sorted_indices = np.argsort(
            predictions
        )[::-1]


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        left, right = st.columns(
            [1, 1]
        )


        # ====================================================
        # IMAGE
        # ====================================================

        with left:

            st.html("""
            <div class="glass-card">

                <div style="
                    font-size:18px;
                    font-weight:800;
                    color:#e2e8f0;
                    margin-bottom:15px;
                ">
                🖼️ INPUT IMAGE
                </div>

            """)

            st.image(
                image,
                use_container_width=True
            )

            st.html("""
            </div>
            """)


        # ====================================================
        # PREDICTION
        # ====================================================

        with right:

            st.html(
                f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                    AI PREDICTION
                    </div>

                    <div class="prediction-name">
                    {predicted_class}
                    </div>

                    <div
                        class="meter"
                        style="
                        --confidence:
                        {confidence:.2f}%;
                        "
                    >

                        <div class="meter-content">

                            <div class="meter-number">
                            {confidence:.1f}%
                            </div>

                            <div class="meter-label">
                            CONFIDENCE
                            </div>

                        </div>

                    </div>

                    <div class="prediction-label">
                    MODEL ANALYSIS COMPLETE
                    </div>

                </div>
                """
            )


        # ====================================================
        # TOP PREDICTIONS
        # ====================================================

        st.html("""
        <div class="section-title">
        🏆 Top Predictions
        </div>

        <div class="section-subtitle">
        The three classes with the highest probability.
        </div>
        """)


        for rank, index in enumerate(
            sorted_indices[:3],
            start=1
        ):

            probability = (
                predictions[index] * 100
            )


            st.html(
                f"""
                <div class="top-card">

                    <span class="rank">
                    #{rank}
                    </span>

                    &nbsp;&nbsp;

                    <span class="class-name">
                    {icons[index]}
                    {class_names[index]}
                    </span>

                    <span class="probability">
                    {probability:.2f}%
                    </span>

                </div>
                """
            )


            st.progress(
                float(
                    predictions[index]
                )
            )


        # ====================================================
        # ALL PROBABILITIES
        # ====================================================

        with st.expander(
            "📊 View all class probabilities"
        ):

            for index in sorted_indices:

                probability = (
                    predictions[index] * 100
                )

                st.write(
                    f"**{icons[index]} "
                    f"{class_names[index]}** — "
                    f"{probability:.2f}%"
                )

                st.progress(
                    float(
                        predictions[index]
                    )
                )


# ============================================================
# MODEL ANALYTICS
# ============================================================

elif page == "📊 Model Analytics":

    st.html("""
    <div class="section-title">
    📊 Model Analytics
    </div>

    <div class="section-subtitle">
    Technical information about the trained classifier.
    </div>
    """)


    a, b = st.columns(2)


    with a:

        st.html("""
        <div class="glass-card">

            <h3>🧠 Neural Network</h3>

            <b>Architecture</b><br>
            Convolutional Neural Network

            <br><br>

            <b>Optimizer</b><br>
            Adam

            <br><br>

            <b>Loss Function</b><br>
            Sparse Categorical Crossentropy

            <br><br>

            <b>Output</b><br>
            10-class Softmax

        </div>
        """)


    with b:

        st.html("""
        <div class="glass-card">

            <h3>📦 Dataset</h3>

            <b>Dataset</b><br>
            CIFAR-10

            <br><br>

            <b>Image Size</b><br>
            32 × 32 RGB

            <br><br>

            <b>Classes</b><br>
            10

            <br><br>

            <b>Test Accuracy</b><br>
            64.19%

        </div>
        """)


    # ========================================================
    # DATA AUGMENTATION
    # ========================================================

    st.html("""
    <div class="section-title">
    🛠️ Data Augmentation
    </div>
    """)


    a, b, c = st.columns(3)


    with a:

        st.html("""
        <div class="info-card">

            <div class="info-icon">
            ↔️
            </div>

            <div class="info-title">
            Horizontal Flip
            </div>

            <div class="info-text">
            Creates horizontally flipped
            training images.
            </div>

        </div>
        """)


    with b:

        st.html("""
        <div class="info-card">

            <div class="info-icon">
            ⟳
            </div>

            <div class="info-title">
            Rotation
            </div>

            <div class="info-text">
            Applies small rotations
            during training.
            </div>

        </div>
        """)


    with c:

        st.html("""
        <div class="info-card">

            <div class="info-icon">
            🔍
            </div>

            <div class="info-title">
            Zoom
            </div>

            <div class="info-text">
            Applies small zoom
            transformations.
            </div>

        </div>
        """)


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "🧠 About the Project":

    st.html("""
    <div class="section-title">
    🧠 About VisionAI
    </div>

    <div class="section-subtitle">
    A deep learning image classification system.
    </div>
    """)


    st.html("""
    <div class="glass-card">

        <h3>
        🚀 Project Overview
        </h3>

        VisionAI is an image classification system built
        using <b>TensorFlow</b>,
        <b>Convolutional Neural Networks</b>
        and the <b>CIFAR-10 dataset</b>.

        <br><br>

        The model processes uploaded images and predicts
        one of ten CIFAR-10 categories.

    </div>
    """)


    st.html("""
    <div class="section-title">
    🎯 Supported Classes
    </div>
    """)


    cols = st.columns(5)


    for i, name in enumerate(
        class_names
    ):

        with cols[i % 5]:

            st.html(
                f"""
                <div class="info-card">

                    <div class="info-icon">
                    {icons[i]}
                    </div>

                    <div class="info-title">
                    {name}
                    </div>

                </div>
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    ◈ VisionAI

    <br><br>

    CIFAR-10 • TensorFlow • CNN • Streamlit

    <br><br>

    Deep Learning Image Classification System

</div>
""")