import streamlit as st
from PIL import Image

# ---------------------- Page Configuration ----------------------
st.set_page_config(
    page_title="Heart Failure Prediction App",
    page_icon="❤️",
    layout="centered"
)

# ---------------------- Main Content ----------------------
st.title("❤️ Heart Failure Prediction App")
st.write("---")

# Optional banner or logo
# image = Image.open("images/heart_banner.png")
# st.image(image, use_container_width=True)

st.markdown("""
### Welcome!

This application helps analyze and predict the likelihood of **heart failure** based on various health parameters such as:
- Age  
- Cholesterol  
- Maximum Heart Rate Achieved  
- ST Depression (Oldpeak)  
- and more...

---

### 🔍 What You Can Do Here:
1. **Go to the “Heart Failure Prediction” page** to input your data and get predictions.  
2. **Explore the “EDA” page** to visualize trends, correlations, and data distributions.

---

### 🧠 About the Model:
This model was trained using **Logistic Regression**, and features were **scaled** using individual scalers for better accuracy.  
It predicts whether a person is likely to have **heart failure (1)** or **not (0)**.

---

💡 *Tip:* Use the sidebar to navigate between pages.
""")

st.write("---")

st.markdown("Developed with ❤️ using **Streamlit** and **Machine Learning**.")
