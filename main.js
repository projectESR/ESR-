// main.js - Main process for Electron

const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pyProc = null;
let mainWindow = null;

// Determine the path to the Python executable
const getPyProcPath = () => {
  if (process.env.NODE_ENV === 'development') {
    // In development, we run the python script directly
    return path.join(__dirname, 'app.py');
  }
  // In production, the python script is packaged into an executable
  // The path will depend on the OS and the pyinstaller build
  if (process.platform === 'win32') {
    return path.join(__dirname, 'dist_py', 'app.exe');
  }
  return path.join(__dirname, 'dist_py', 'app');
};

const createPyProc = () => {
  let script = getPyProcPath();
  
  if (process.env.NODE_ENV === 'development') {
      pyProc = spawn('python', [script]);
  } else {
      pyProc = spawn(script);
  }

  pyProc.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
  });

  pyProc.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
  });
};

const createMainWindow = () => {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Load the Flask app's URL
  mainWindow.loadURL('http://127.0.0.1:5000');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
};

app.on('ready', () => {
  createPyProc();
  setTimeout(createMainWindow, 3000); // Give python server time to start
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    // Kill the python process before quitting
    if (pyProc) {
      pyProc.kill();
      pyProc = null;
    }
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createMainWindow();
  }
});
