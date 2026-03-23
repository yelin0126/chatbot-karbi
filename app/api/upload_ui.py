"""
Chat UI with model selector and chat history.
Access at: http://localhost:8000/ui
"""

import logging
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger("tilon.ui")
router = APIRouter(tags=["Chat UI"])
UI_INDEX_PATH = Path(__file__).resolve().parents[2] / "static" / "index.html"

CHAT_UI_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tilon AI Chatbot</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f1117;color:#e4e4e7;height:100vh;display:flex}

        /* ── Sidebar (Chat History) ── */
        .sidebar{width:300px;background:#18181b;border-right:1px solid #27272a;display:flex;flex-direction:column;flex-shrink:0}
        .sidebar-top{padding:12px}
        .new-chat-btn{width:100%;padding:10px;border-radius:8px;border:1px solid #3f3f46;background:transparent;color:#e4e4e7;font-size:.85rem;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px}
        .new-chat-btn:hover{background:#27272a}
        .history-list{flex:1;overflow-y:auto;padding:4px 8px}
        .history-item{padding:10px 12px;border-radius:8px;font-size:.8rem;color:#a1a1aa;cursor:pointer;margin-bottom:2px;display:flex;justify-content:space-between;align-items:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .history-item:hover{background:#27272a}
        .history-item.active{background:#27272a;color:#fff}
        .history-item .del{display:none;background:none;border:none;color:#71717a;cursor:pointer;font-size:.9rem;padding:0 2px;flex-shrink:0}
        .history-item:hover .del{display:block}
        .history-item .del:hover{color:#f87171}
        .sidebar-section-title{padding:10px 14px 6px;font-size:.68rem;color:#71717a;text-transform:uppercase;letter-spacing:.08em}
        .upload-list{max-height:34vh;overflow-y:auto;padding:0 8px 8px}
        .upload-item{width:100%;padding:10px 12px;border-radius:8px;border:1px solid transparent;background:transparent;color:#d4d4d8;cursor:pointer;text-align:left;margin-bottom:4px}
        .upload-item:hover{background:#27272a}
        .upload-item.active-upload{background:#1e1b4b;border-color:#312e81}
        .upload-name{display:block;font-size:.78rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .upload-meta{display:block;margin-top:4px;font-size:.66rem;color:#94a3b8}
        .upload-empty{padding:10px 14px 14px;font-size:.74rem;color:#52525b}
        .sidebar-actions{padding:0 12px 10px;display:flex;gap:6px}
        .sidebar-btn{flex:1;padding:8px 10px;border-radius:8px;border:1px solid #3f3f46;background:transparent;color:#a1a1aa;font-size:.76rem;cursor:pointer}
        .sidebar-btn:hover{background:#27272a;color:#fff}
        .sidebar-footer{padding:10px 12px;border-top:1px solid #27272a;font-size:.7rem;color:#3f3f46;text-align:center}

        /* ── Main ── */
        .main{flex:1;display:flex;flex-direction:column;min-width:0}

        /* Topbar */
        .topbar{padding:8px 20px;border-bottom:1px solid #27272a;display:flex;align-items:center;justify-content:space-between;background:#18181b;flex-shrink:0;gap:12px}
        .topbar h1{font-size:.95rem;color:#fff;white-space:nowrap}
        .topbar-center{display:flex;align-items:center;gap:10px}
        .model-select{padding:5px 10px;border-radius:6px;border:1px solid #3f3f46;background:#27272a;color:#e4e4e7;font-size:.78rem;outline:none;cursor:pointer}
        .model-select:focus{border-color:#6366f1}
        .topbar-right{display:flex;align-items:center;gap:10px;font-size:.72rem}
        .status-badge{display:flex;align-items:center;gap:3px;color:#71717a}
        .dot{width:6px;height:6px;border-radius:50%}.dot.green{background:#22c55e}.dot.red{background:#ef4444}
        .topbar-btn{padding:4px 8px;border-radius:5px;border:1px solid #3f3f46;background:transparent;color:#a1a1aa;font-size:.7rem;cursor:pointer}
        .topbar-btn:hover{background:#27272a;color:#fff}

        /* Messages */
        .messages{flex:1;overflow-y:auto;padding:20px 16px;scroll-behavior:smooth}
        .msg-wrap{max-width:800px;margin:0 auto 16px}
        .message{display:flex;gap:10px}
        .message .avatar{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:.75rem;flex-shrink:0;font-weight:700}
        .message.user .avatar{background:#3730a3;color:#c7d2fe}
        .message.assistant .avatar{background:#065f46;color:#6ee7b7}
        .message.system .avatar{background:#422006;color:#fcd34d;font-size:.65rem}
        .message .body{flex:1;min-width:0}
        .message .sender{font-size:.7rem;color:#52525b;margin-bottom:2px;font-weight:600;display:flex;align-items:center;gap:5px}
        .message .content{font-size:.86rem;line-height:1.6;color:#d4d4d8;white-space:pre-wrap;word-break:break-word}
        .message.assistant .content{background:#1c1c22;padding:12px 16px;border-radius:10px;border:1px solid #27272a}
        .mode-tag{font-size:.58rem;padding:1px 4px;background:#052e16;color:#86efac;border-radius:3px}
        .sources{margin-top:6px;display:flex;flex-wrap:wrap;gap:4px}
        .source-tag{font-size:.66rem;padding:2px 7px;background:#1e1b4b;color:#a5b4fc;border-radius:4px;border:1px solid #312e81}
        .file-badge{font-size:.72rem;padding:4px 8px;background:#1e1b4b;color:#c7d2fe;border-radius:5px;margin-bottom:6px;display:inline-block;border:1px solid #312e81}

        /* Typing */
        .typing{display:none;max-width:800px;margin:0 auto;padding:0 16px}
        .typing.active{display:flex;gap:10px}
        .typing .avatar{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:.75rem;flex-shrink:0;background:#065f46;color:#6ee7b7;font-weight:700}
        .typing-dots{display:flex;gap:4px;padding:10px 14px;background:#1c1c22;border-radius:10px;border:1px solid #27272a}
        .typing-dots span{width:6px;height:6px;background:#52525b;border-radius:50%;animation:bounce 1.4s infinite ease-in-out}
        .typing-dots span:nth-child(2){animation-delay:.2s}
        .typing-dots span:nth-child(3){animation-delay:.4s}
        @keyframes bounce{0%,80%,100%{transform:scale(.8);opacity:.4}40%{transform:scale(1);opacity:1}}

        /* Input */
        .input-area{padding:10px 16px 14px;border-top:1px solid #27272a;background:#18181b;flex-shrink:0}
        .input-container{max-width:800px;margin:0 auto}
        .attached-file{display:none;align-items:center;gap:6px;padding:6px 10px;background:#1e1b4b;border:1px solid #312e81;border-radius:7px;margin-bottom:6px;font-size:.78rem;color:#c7d2fe}
        .attached-file.visible{display:flex}
        .attached-file .af-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .attached-file .af-size{color:#6366f1;font-size:.68rem}
        .attached-file .af-remove{background:none;border:none;color:#f87171;cursor:pointer;font-size:1rem;padding:0 2px}
        .active-scope{display:none;align-items:center;gap:8px;padding:6px 10px;background:#052e16;border:1px solid #14532d;border-radius:7px;margin-bottom:6px;font-size:.78rem;color:#bbf7d0}
        .active-scope.visible{display:flex}
        .active-scope .scope-label{font-size:.68rem;color:#86efac;text-transform:uppercase;letter-spacing:.04em}
        .active-scope .scope-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .active-scope .scope-clear{background:none;border:none;color:#fca5a5;cursor:pointer;font-size:1rem;padding:0 2px}
        .input-row{display:flex;gap:6px;align-items:flex-end}
        .attach-btn{width:40px;height:40px;border-radius:8px;border:1px solid #3f3f46;background:transparent;color:#71717a;font-size:1.1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}
        .attach-btn:hover{background:#27272a;color:#a5b4fc;border-color:#6366f1}
        .attach-btn.has-file{background:#1e1b4b;color:#a5b4fc;border-color:#6366f1}
        .input-row textarea{flex:1;padding:9px 12px;background:#27272a;border:1px solid #3f3f46;border-radius:8px;color:#e4e4e7;font-size:.86rem;font-family:inherit;resize:none;min-height:40px;max-height:140px;outline:none;line-height:1.4}
        .input-row textarea:focus{border-color:#6366f1}
        .input-row textarea::placeholder{color:#52525b}
        .send-btn{width:40px;height:40px;border-radius:8px;border:none;background:#6366f1;color:#fff;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0}
        .send-btn:hover{background:#4f46e5}
        .send-btn:disabled{background:#3f3f46;cursor:not-allowed}
        input[type="file"]{display:none}

        /* Docs drawer */
        .drawer-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:100}
        .drawer-overlay.open{display:block}
        .drawer{position:fixed;right:-360px;top:0;bottom:0;width:320px;background:#18181b;border-left:1px solid #27272a;z-index:101;transition:right .25s;display:flex;flex-direction:column}
        .drawer.open{right:0}
        .drawer-header{padding:14px 18px;border-bottom:1px solid #27272a;display:flex;justify-content:space-between;align-items:center}
        .drawer-header h3{font-size:.9rem;color:#fff}
        .drawer-close{background:none;border:none;color:#71717a;font-size:1.1rem;cursor:pointer}
        .drawer-body{flex:1;overflow-y:auto;padding:12px}
        .drawer-doc{padding:7px 10px;background:#0f1117;border-radius:5px;margin-bottom:3px;font-size:.76rem;display:flex;justify-content:space-between}
        .drawer-doc .name{color:#d4d4d8;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .drawer-doc .chunks{color:#6366f1;font-weight:600;margin-left:8px}
        .drawer-empty{color:#3f3f46;font-size:.78rem;text-align:center;padding:24px}
        .drawer-actions{padding:10px 14px;border-top:1px solid #27272a;display:flex;gap:6px}
        .drawer-actions button{flex:1;padding:6px;border-radius:5px;border:none;cursor:pointer;font-size:.72rem;font-weight:500}
        .btn-ref{background:#27272a;color:#a1a1aa}.btn-ref:hover{background:#3f3f46;color:#fff}
        .btn-rst{background:#450a0a;color:#fca5a5}.btn-rst:hover{background:#7f1d1d}

        @media(max-width:768px){.sidebar{display:none}}
    </style>
</head>
<body>
    <!-- Sidebar: Chat History -->
    <aside class="sidebar">
        <div class="sidebar-top">
            <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
        </div>
        <div class="sidebar-section-title">Chats</div>
        <div class="history-list" id="historyList"></div>
        <div class="sidebar-section-title">Uploaded Files</div>
        <div class="upload-list" id="uploadList"><div class="upload-empty">No uploaded files yet</div></div>
        <div class="sidebar-actions">
            <button class="sidebar-btn" onclick="loadUploads()">Refresh Uploads</button>
            <button class="sidebar-btn" onclick="clearActiveSource()">Clear Selection</button>
        </div>
        <div class="sidebar-footer">Tilon AI Chatbot v7.4</div>
    </aside>

    <main class="main">
        <!-- Topbar -->
        <div class="topbar">
            <h1>Tilon AI</h1>
            <div class="topbar-center">
                <select class="model-select" id="modelSelect"></select>
            </div>
            <div class="topbar-right">
                <span class="status-badge"><span class="dot" id="ollamaDot"></span><span id="ollamaStatus">...</span></span>
                <span class="status-badge" id="chunksBadge">0</span>
                <button class="topbar-btn" onclick="toggleDrawer()">Docs</button>
            </div>
        </div>

        <!-- Messages -->
        <div class="messages" id="messages"></div>

        <!-- Typing -->
        <div class="typing" id="typingIndicator">
            <div class="avatar">AI</div>
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>

        <!-- Input -->
        <div class="input-area">
            <div class="input-container">
                <div class="attached-file" id="attachedFile">
                    <span id="afName"></span>
                    <span class="af-size" id="afSize"></span>
                    <button class="af-remove" onclick="clearFiles()">&times;</button>
                </div>
                <div class="active-scope" id="activeScope">
                    <span class="scope-label">Scoped to</span>
                    <span class="scope-name" id="activeScopeName"></span>
                    <button class="scope-clear" onclick="clearActiveSource()">&times;</button>
                </div>
                <div class="input-row">
                    <button class="attach-btn" id="attachBtn" onclick="document.getElementById('fileInput').click()">&#128206;</button>
                    <textarea id="chatInput" placeholder="Type a message or attach a file..." rows="1"
                        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}"></textarea>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">&#10148;</button>
                </div>
            </div>
            <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.webp" multiple>
        </div>
    </main>

    <!-- Docs Drawer -->
    <div class="drawer-overlay" id="drawerOverlay" onclick="toggleDrawer()"></div>
    <div class="drawer" id="drawer">
        <div class="drawer-header"><h3>Stored Documents</h3><button class="drawer-close" onclick="toggleDrawer()">&times;</button></div>
        <div class="drawer-body" id="docList"><div class="drawer-empty">No documents</div></div>
        <div class="drawer-actions"><button class="btn-ref" onclick="loadDocs()">Refresh</button><button class="btn-rst" onclick="resetDB()">Reset DB</button></div>
    </div>

<script>
const messagesEl=document.getElementById('messages');
const chatInput=document.getElementById('chatInput');
const sendBtn=document.getElementById('sendBtn');
const attachBtn=document.getElementById('attachBtn');
const typingEl=document.getElementById('typingIndicator');
const fileInput=document.getElementById('fileInput');
const attachedFileEl=document.getElementById('attachedFile');
const afName=document.getElementById('afName');
const afSize=document.getElementById('afSize');
const activeScopeEl=document.getElementById('activeScope');
const activeScopeName=document.getElementById('activeScopeName');
const modelSelect=document.getElementById('modelSelect');
const historyList=document.getElementById('historyList');
const uploadList=document.getElementById('uploadList');

let pendingFiles=[];
let rememberedUploads=[];
let currentChatId=null;
let chats={};  // {id: {title, messages: [{role,content,sources,mode,fileName}]}}

// ═══════════════════════════════════════════════════════════
// Chat History (localStorage)
// ═══════════════════════════════════════════════════════════

function loadChats(){
    try{chats=JSON.parse(localStorage.getItem('tilon_chats')||'{}');}catch{chats={};}
    for(const id of Object.keys(chats)){
        if(!Array.isArray(chats[id].messages))chats[id].messages=[];
        if(typeof chats[id].activeSource!=='string')chats[id].activeSource='';
        if(typeof chats[id].activeDocId!=='string')chats[id].activeDocId='';
        if(!Array.isArray(chats[id].activeDocs)){
            chats[id].activeDocs=chats[id].activeSource||chats[id].activeDocId
                ? [{source:chats[id].activeSource||'',docId:chats[id].activeDocId||''}]
                : [];
        }
    }
}
function saveChats(){
    try{localStorage.setItem('tilon_chats',JSON.stringify(chats));}catch{}
}

function newChat(){
    currentChatId='chat_'+Date.now();
    chats[currentChatId]={title:'New Chat',messages:[],activeSource:'',activeDocId:'',activeDocs:[]};
    saveChats();
    messagesEl.innerHTML='';
    renderActiveSource();
    renderHistory();
    chatInput.focus();
}

function loadChat(id){
    currentChatId=id;
    const chat=chats[id];
    if(!chat)return;
    messagesEl.innerHTML='';
    for(const m of chat.messages){
        appendMessageDOM(m.role,m.content,m.sources,m.mode,m.fileName);
    }
    renderActiveSource();
    renderHistory();
    scrollBottom();
}

function deleteChat(id,e){
    e.stopPropagation();
    delete chats[id];
    saveChats();
    if(currentChatId===id)newChat();
    else renderHistory();
}

function renderHistory(){
    const sorted=Object.entries(chats).sort((a,b)=>parseInt(b[0].split('_')[1])-parseInt(a[0].split('_')[1]));
    historyList.innerHTML='';
    for(const[id,chat]of sorted){
        const div=document.createElement('div');
        div.className='history-item'+(id===currentChatId?' active':'');
        div.onclick=()=>loadChat(id);
        div.innerHTML=`<span style="flex:1;overflow:hidden;text-overflow:ellipsis">${esc(chat.title)}</span><button class="del" onclick="deleteChat('${id}',event)">&times;</button>`;
        historyList.appendChild(div);
    }
}

function updateChatTitle(text){
    if(!currentChatId||!chats[currentChatId])return;
    if(chats[currentChatId].title==='New Chat'){
        chats[currentChatId].title=text.slice(0,40)+(text.length>40?'...':'');
        saveChats();renderHistory();
    }
}

function renderActiveSource(){
    const activeDocs=(currentChatId&&chats[currentChatId])?chats[currentChatId].activeDocs||[]:[];
    if(activeDocs.length===1){
        activeScopeName.textContent=activeDocs[0].source||activeDocs[0].docId||'Selected document';
        activeScopeEl.classList.add('visible');
    }else if(activeDocs.length>1){
        activeScopeName.textContent=`${activeDocs.length} documents selected for comparison`;
        activeScopeEl.classList.add('visible');
    }else{
        activeScopeName.textContent='';
        activeScopeEl.classList.remove('visible');
    }
    renderUploads();
}

function setActiveDocuments(docs){
    if(!currentChatId||!chats[currentChatId])return;
    const normalized=(docs||[]).filter(doc=>doc&&(doc.source||doc.docId)).map(doc=>({
        source:doc.source||'',
        docId:doc.docId||''
    }));
    chats[currentChatId].activeDocs=normalized;
    chats[currentChatId].activeSource=normalized.length===1?(normalized[0].source||''):'';
    chats[currentChatId].activeDocId=normalized.length===1?(normalized[0].docId||''):'';
    saveChats();
    renderActiveSource();
}

function setActiveDocument(source,docId){
    setActiveDocuments(source||docId?[{source:source||'',docId:docId||''}]:[]);
}

function toggleActiveDocument(source,docId){
    if(!currentChatId||!chats[currentChatId])return;
    const activeDocs=chats[currentChatId].activeDocs||[];
    const exists=activeDocs.some(doc=>(doc.docId&&doc.docId===docId)||(!doc.docId&&doc.source===source));
    if(exists){
        setActiveDocuments(activeDocs.filter(doc=>!((doc.docId&&doc.docId===docId)||(!doc.docId&&doc.source===source))));
    }else{
        setActiveDocuments(activeDocs.concat([{source:source||'',docId:docId||''}]));
    }
}

function clearActiveSource(){
    if(!currentChatId||!chats[currentChatId]||!(chats[currentChatId].activeDocs||[]).length)return;
    const cleared=(chats[currentChatId].activeDocs||[]).map(doc=>doc.source||doc.docId).join(', ');
    setActiveDocuments([]);
    appendMessageDOM('system','Document selection cleared: '+cleared);
    pushMessage('system','Document selection cleared: '+cleared);
}

function pushMessage(role,content,sources,mode,fileName){
    if(!currentChatId)newChat();
    chats[currentChatId].messages.push({role,content,sources:sources||[],mode:mode||'',fileName:fileName||''});
    saveChats();
}

// ═══════════════════════════════════════════════════════════
// File Attachment
// ═══════════════════════════════════════════════════════════

fileInput.addEventListener('change',()=>{
    if(fileInput.files.length>0){
        const seen=new Set(pendingFiles.map(file=>`${file.name}::${file.size}::${file.lastModified}`));
        for(const file of Array.from(fileInput.files)){
            const key=`${file.name}::${file.size}::${file.lastModified}`;
            if(!seen.has(key)){
                pendingFiles.push(file);
                seen.add(key);
            }
        }
        renderPendingFiles();
        chatInput.focus();
    }
    fileInput.value='';
});

function renderPendingFiles(){
    if(!pendingFiles.length){
        afName.textContent='';
        afSize.textContent='';
        attachedFileEl.classList.remove('visible');
        attachBtn.classList.remove('has-file');
        return;
    }
    const totalBytes=pendingFiles.reduce((sum,file)=>sum+file.size,0);
    const names=pendingFiles.slice(0,2).map(file=>file.name);
    afName.textContent=pendingFiles.length>2
        ? `${names.join(', ')} +${pendingFiles.length-2} more`
        : names.join(', ');
    afSize.textContent=`${pendingFiles.length} file${pendingFiles.length>1?'s':''} • ${fmtBytes(totalBytes)}`;
    attachedFileEl.classList.add('visible');
    attachBtn.classList.add('has-file');
}

function clearFiles(){pendingFiles=[];renderPendingFiles();}
function fmtBytes(b){if(b<1024)return b+' B';if(b<1048576)return(b/1024).toFixed(1)+' KB';return(b/1048576).toFixed(1)+' MB';}

// ═══════════════════════════════════════════════════════════
// Send Message
// ═══════════════════════════════════════════════════════════

async function sendMessage(){
    const text=chatInput.value.trim();
    const files=pendingFiles.slice();
    if(!text&&!files.length)return;
    if(!currentChatId)newChat();

    const displayText=text||(files.length===1?'Analyze this document':`Upload ${files.length} documents`);
    const selectedModel=modelSelect.value;
    const activeDocs=chats[currentChatId].activeDocs||[];
    const activeSource=activeDocs.length===1?(activeDocs[0].source||''):'';
    const activeDocId=activeDocs.length===1?(activeDocs[0].docId||''):'';
    const activeSources=activeDocs.map(doc=>doc.source).filter(Boolean);
    const activeDocIds=activeDocs.map(doc=>doc.docId).filter(Boolean);

    appendMessageDOM(
        'user',
        displayText,
        null,
        null,
        files.length===1?files[0].name:(files.length>1?`${files.length} files attached`:null)
    );
    pushMessage(
        'user',
        displayText,
        null,
        null,
        files.length===1?files[0].name:(files.length>1?`${files.length} files attached`:null)
    );
    updateChatTitle(displayText);

    chatInput.value='';chatInput.style.height='auto';
    clearFiles();sendBtn.disabled=true;typingEl.classList.add('active');scrollBottom();

    try{
        let data;
        if(files.length===1){
            const file=files[0];
            const fd=new FormData();
            fd.append('file',file);
            fd.append('message',displayText);
            fd.append('model',selectedModel);
            const resp=await fetch('/chat-with-file',{method:'POST',body:fd});
            data=await resp.json();
            if(!resp.ok){showError(data.detail||'Upload failed');return;}
            const responseDocs=(data.active_doc_ids||[]).map((docId,idx)=>({
                source:(data.active_sources||[])[idx]||data.active_source||file.name,
                docId:docId||''
            }));
            if(responseDocs.length){setActiveDocuments(responseDocs);}
            else{setActiveDocument(data.active_source||file.name,data.active_doc_id||'');}
            if(data.ingest&&data.ingest.count>0){
                appendMessageDOM('system',file.name+' — '+data.ingest.count+' chunks ingested');
                pushMessage('system',file.name+' — '+data.ingest.count+' chunks ingested');
            }
            await loadUploads();
            await loadDocs();
        }else if(files.length>1){
            const fd=new FormData();
            for(const file of files)fd.append('files',file);
            const resp=await fetch('/upload-multiple',{method:'POST',body:fd});
            data=await resp.json();
            if(!resp.ok){showError(data.detail||'Multi-upload failed');return;}
            const successes=(data.results||[]).filter(item=>item.status==='success');
            const failures=(data.results||[]).filter(item=>item.status!=='success');
            let summary=`Uploaded ${successes.length} file${successes.length===1?'':'s'} successfully.`;
            if(failures.length)summary+=` ${failures.length} failed or were skipped.`;
            if(text)summary+=' Multiple files were uploaded, so no single document was auto-selected. Pick one from the sidebar and ask your question again.';
            appendMessageDOM('system',summary);
            pushMessage('system',summary);
            await loadUploads();
            await loadDocs();
            loadHealth();
            return;
        }else{
            const history=chats[currentChatId].messages.filter(m=>m.role==='user'||m.role==='assistant').slice(-8);
            const resp=await fetch('/chat',{
                method:'POST',headers:{'Content-Type':'application/json'},
                body:JSON.stringify({
                    message:text,
                    history:history,
                    model:selectedModel,
                    active_source:activeSource||null,
                    active_doc_id:activeDocId||null,
                    active_sources:activeSources,
                    active_doc_ids:activeDocIds
                })
            });
            data=await resp.json();
            if(!resp.ok){showError(data.detail||'Error');return;}
            if(
                Object.prototype.hasOwnProperty.call(data,'active_source')
                || Object.prototype.hasOwnProperty.call(data,'active_doc_id')
                || Object.prototype.hasOwnProperty.call(data,'active_sources')
                || Object.prototype.hasOwnProperty.call(data,'active_doc_ids')
            ){
                const responseDocs=(data.active_doc_ids||[]).map((docId,idx)=>({
                    source:(data.active_sources||[])[idx]||'',
                    docId:docId||''
                })).filter(doc=>doc.source||doc.docId);
                if(responseDocs.length){setActiveDocuments(responseDocs);}
                else{setActiveDocument(data.active_source||'',data.active_doc_id||'');}
            }
        }
        appendMessageDOM('assistant',data.answer,data.sources,data.mode);
        pushMessage('assistant',data.answer,data.sources,data.mode);
        loadHealth();
    }catch(err){showError('Connection error: '+err.message);}
    finally{typingEl.classList.remove('active');sendBtn.disabled=false;chatInput.focus();scrollBottom();}
}

function showError(msg){
    appendMessageDOM('assistant','Error: '+msg);
    pushMessage('assistant','Error: '+msg);
    typingEl.classList.remove('active');sendBtn.disabled=false;
}

// ═══════════════════════════════════════════════════════════
// Render
// ═══════════════════════════════════════════════════════════

function appendMessageDOM(role,content,sources,mode,fileName){
    const av={user:'U',assistant:'AI',system:'i'};
    const nm={user:'You',assistant:'Tilon AI',system:'System'};
    let fileHtml=fileName?`<div class="file-badge">&#128206; ${esc(fileName)}</div>`:'';
    let modeHtml=mode?`<span class="mode-tag">${esc(mode)}</span>`:'';
    let srcHtml='';
    if(sources&&sources.length){
        srcHtml='<div class="sources">'+sources.map(s=>{
            const label=`${s.source||'?'} p.${s.page||'?'}`;
            const title=[s.doc_id||'',s.source_type||''].filter(Boolean).join(' | ');
            return `<span class="source-tag" title="${esc(title)}">${esc(label)}</span>`;
        }).join('')+'</div>';
    }
    const w=document.createElement('div');w.className='msg-wrap';
    w.innerHTML=`<div class="message ${role}"><div class="avatar">${av[role]||'?'}</div><div class="body"><div class="sender">${nm[role]||role} ${modeHtml}</div>${fileHtml}<div class="content">${esc(content)}</div>${srcHtml}</div></div>`;
    messagesEl.appendChild(w);scrollBottom();
}

function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML;}
function scrollBottom(){setTimeout(()=>{messagesEl.scrollTop=messagesEl.scrollHeight;},50);}
chatInput.addEventListener('input',()=>{chatInput.style.height='auto';chatInput.style.height=Math.min(chatInput.scrollHeight,140)+'px';});

// ═══════════════════════════════════════════════════════════
// Models
// ═══════════════════════════════════════════════════════════

async function loadModels(){
    try{
        const resp=await fetch('/models');
        const data=await resp.json();
        modelSelect.innerHTML='';
        const saved=localStorage.getItem('tilon_model');
        for(const m of(data.available||[])){
            const opt=document.createElement('option');
            opt.value=m.trim();opt.textContent=m.trim();
            if(saved&&m.trim()===saved)opt.selected=true;
            else if(!saved&&m.trim()===data.default)opt.selected=true;
            modelSelect.appendChild(opt);
        }
    }catch{modelSelect.innerHTML='<option>llama3.1:latest</option>';}
}

modelSelect.addEventListener('change',()=>{localStorage.setItem('tilon_model',modelSelect.value);});

// ═══════════════════════════════════════════════════════════
// Remembered Uploads
// ═══════════════════════════════════════════════════════════

function renderUploads(){
    if(!rememberedUploads.length){
        uploadList.innerHTML='<div class="upload-empty">No uploaded files yet</div>';
        return;
    }
    const activeDocs=(currentChatId&&chats[currentChatId])?chats[currentChatId].activeDocs||[]:[];
    uploadList.innerHTML='';
    for(const doc of rememberedUploads){
        const button=document.createElement('button');
        const isSelected=activeDocs.some(active=>active.docId===doc.doc_id||(!active.docId&&active.source===doc.source));
        button.className='upload-item'+(isSelected?' active-upload':'');
        button.onclick=()=>selectRememberedUpload(doc);
        const meta=[
            doc.page_total?`${doc.page_total}p`:'',
            doc.chunk_count?`${doc.chunk_count} chunks`:'',
            Array.isArray(doc.languages)&&doc.languages.length?doc.languages.join('/'):''
        ].filter(Boolean).join(' • ');
        button.innerHTML=`<span class="upload-name">${esc(doc.source||'Unnamed upload')}</span><span class="upload-meta">${esc(meta||'upload')}</span>`;
        uploadList.appendChild(button);
    }
}

function selectRememberedUpload(doc){
    if(!currentChatId)newChat();
    const wasSelected=(chats[currentChatId].activeDocs||[]).some(active=>active.docId===doc.doc_id||(!active.docId&&active.source===doc.source));
    toggleActiveDocument(doc.source||'',doc.doc_id||'');
    const activeCount=(chats[currentChatId].activeDocs||[]).length;
    let note=`Document scope set from sidebar: ${doc.source||'Unnamed upload'}`;
    if(wasSelected){
        note=activeCount
            ? `Removed from comparison scope: ${doc.source||'Unnamed upload'} (${activeCount} still selected)`
            : `Document selection cleared for ${doc.source||'Unnamed upload'}`;
    }else if(activeCount>1){
        note=`Added to comparison scope: ${doc.source||'Unnamed upload'} (${activeCount} selected)`;
    }
    appendMessageDOM('system',note);
    pushMessage('system',note);
}

async function loadUploads(){
    try{
        const resp=await fetch('/uploaded-docs');
        const data=await resp.json();
        rememberedUploads=Array.isArray(data.documents)?data.documents:[];
        renderUploads();
    }catch{
        rememberedUploads=[];
        uploadList.innerHTML='<div class="upload-empty">Could not load uploaded files</div>';
    }
}

// ═══════════════════════════════════════════════════════════
// Docs Drawer
// ═══════════════════════════════════════════════════════════

function toggleDrawer(){document.getElementById('drawer').classList.toggle('open');document.getElementById('drawerOverlay').classList.toggle('open');loadDocs();}

async function loadDocs(){
    const dl=document.getElementById('docList');
    try{
        const resp=await fetch('/docs-list');const data=await resp.json();
        if(!data.documents||!data.documents.length){dl.innerHTML='<div class="drawer-empty">No documents stored</div>';return;}
        const g={};
        for(const d of data.documents){
            const key=d.doc_id||`${d.source||'?'}::${d.source_type||''}`;
            if(!g[key])g[key]={source:d.source||'?',sourceType:d.source_type||'',chunks:0,pageTotal:d.page_total||''};
            g[key].chunks++;
        }
        dl.innerHTML='';
        for(const item of Object.values(g)){
            const subtitle=[item.sourceType,item.pageTotal?`${item.pageTotal}p`:'' ].filter(Boolean).join(' • ');
            const d=document.createElement('div');
            d.className='drawer-doc';
            d.innerHTML=`<div style="display:flex;flex-direction:column;min-width:0;flex:1"><span class="name">${esc(item.source)}</span>${subtitle?`<span style="font-size:.68rem;color:#94a3b8">${esc(subtitle)}</span>`:''}</div><span class="chunks">${item.chunks}</span>`;
            dl.appendChild(d);
        }
    }catch{dl.innerHTML='<div class="drawer-empty">Error</div>';}
}

async function resetDB(){
    if(!confirm('Reset vector DB? All documents deleted.'))return;
    try{
        await fetch('/reset-db',{method:'DELETE'});
        for(const id of Object.keys(chats)){
            chats[id].activeSource='';
            chats[id].activeDocId='';
        }
        saveChats();
        renderActiveSource();
        appendMessageDOM('system','Vector DB reset.');
        pushMessage('system','Vector DB reset.');
        loadDocs();loadUploads();loadHealth();
    }
    catch(err){appendMessageDOM('system','Reset failed');}
}

// ═══════════════════════════════════════════════════════════
// Health
// ═══════════════════════════════════════════════════════════

async function loadHealth(){
    try{
        const resp=await fetch('/health');const d=await resp.json();
        document.getElementById('ollamaDot').className='dot '+(d.ollama==='connected'?'green':'red');
        document.getElementById('ollamaStatus').textContent=d.ollama==='connected'?'Online':'Offline';
        document.getElementById('chunksBadge').textContent=(d.documents_in_vectorstore||0)+' chunks';
    }catch{document.getElementById('ollamaDot').className='dot red';document.getElementById('ollamaStatus').textContent='Err';}
}

// ═══════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════

loadChats();
loadModels();
loadHealth();
loadUploads();
setInterval(loadHealth,30000);

// Load most recent chat or start new
const chatIds=Object.keys(chats).sort((a,b)=>parseInt(b.split('_')[1])-parseInt(a.split('_')[1]));
if(chatIds.length>0){loadChat(chatIds[0]);}else{newChat();}

chatInput.focus();
</script>
</body>
</html>
"""

@router.get("/ui", response_class=HTMLResponse)
def chat_ui():
    if UI_INDEX_PATH.exists():
        return FileResponse(UI_INDEX_PATH)
    logger.warning("static UI not found at %s, serving embedded fallback UI", UI_INDEX_PATH)
    return HTMLResponse(CHAT_UI_HTML)
