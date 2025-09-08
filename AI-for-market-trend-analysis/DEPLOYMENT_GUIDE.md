# Deployment Guide for AI Market Trends Analyzer

This guide provides step-by-step instructions to deploy the AI Market Trends Analyzer Streamlit application on a local machine or server.

---

## Prerequisites

- Python 3.8 or higher installed on your system.
- Git installed (optional, for cloning the repository).
- Internet connection for downloading dependencies and fetching stock data.

---

## Setup Instructions

### 1. Clone the Repository (Optional)

If you have not already cloned the project repository, run:

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

Start the Streamlit app with the following command:

```bash
python -m streamlit run streamlit_app_complete.py
```

This will launch the app locally. You should see output similar to:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8505
Network URL: http://<your-ip>:8505
```

Open the Local URL in your web browser to access the app.

---

## Notes and Troubleshooting

- If you encounter the error `'streamlit' is not recognized as an internal or external command`, ensure you are running the command inside the activated virtual environment or use `python -m streamlit` as shown above.
- The app fetches live stock data from external APIs; ensure your machine has internet access.
- If some stock symbols return errors or no data, they may be delisted or unavailable in the data source.
- To stop the app, press `Ctrl+C` in the terminal where the app is running.

---

## Optional: Running on a Server

- For deployment on a cloud server or VPS, ensure Python and dependencies are installed as above.
- Use tools like `tmux` or `screen` to run the app in a persistent session.
- Configure firewall rules to allow incoming traffic on the Streamlit port (default 8505).
- Consider using a reverse proxy (e.g., Nginx) for SSL termination and domain routing.

---

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

---

This completes the deployment guide for the AI Market Trends Analyzer application.
