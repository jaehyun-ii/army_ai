const { contextBridge, ipcRenderer } = require('electron')

// Electron API를 렌더러 프로세스에 안전하게 노출
contextBridge.exposeInMainWorld('electronAPI', {
  // 앱 정보
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),

  // 다이얼로그
  showMessageBox: (options) => ipcRenderer.invoke('show-message-box', options),
  showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
  showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),

  // 카메라 권한
  requestCameraPermission: () => ipcRenderer.invoke('request-camera-permission'),

  // GPU 정보
  getGpuInfo: () => ipcRenderer.invoke('get-gpu-info'),

  // 플랫폼 정보
  platform: process.platform,

  // 앱 이벤트
  onAppReady: (callback) => ipcRenderer.on('app-ready', callback),
  onAppClose: (callback) => ipcRenderer.on('app-close', callback),

  // 파일 시스템 (보안상 제한적으로 노출)
  isElectron: true
})

// 렌더러 프로세스가 Node.js API에 직접 접근하는 것을 방지
window.addEventListener('DOMContentLoaded', () => {
  // 개발 환경에서만 콘솔에 정보 출력
  if (process.env.NODE_ENV === 'development') {
    console.log('Electron preload script loaded')
    console.log('Platform:', process.platform)
    console.log('Electron version:', process.versions.electron)
  }
})