# Tilon AI Chatbot Test Cases

This test pack is based on the current implemented behavior in:

- `app/api/routes.py`
- `app/api/openai_compat.py`
- `app/api/upload_ui.py`
- `app/chat/handlers.py`
- `app/core/document_registry.py`

## 1. Test Scope

The chatbot currently supports:

- general chat through `/chat`
- file upload + ask in one step through `/chat-with-file`
- single and multi-file upload through `/upload` and `/upload-multiple`
- upload-scoped follow-up chat
- remembered uploaded documents in the UI sidebar
- multi-document comparison scope
- OCR/direct extraction flow for images and scanned docs
- document list, health, model list, and reset endpoints
- OpenAI-compatible `/v1/...` endpoints
- optional web search path when configured

## 2. Suggested Test Data

Prepare at least these files before testing:

- one clean text PDF
- one scanned PDF
- one screenshot or image with readable text
- one PDF with content that does not contain the asked answer
- two related PDFs for comparison
- one bundled PDF containing multiple internal sections/guidelines

## 3. Core Use Cases

### UC-01 General Chat

- User opens the UI and asks a normal non-document question.
- Expected:
  - the system returns an answer without requiring file upload
  - the response appears in chat history
  - no document scope is set

### UC-02 Upload One Document And Ask Immediately

- User uploads one PDF and asks a question in the same action.
- Expected:
  - file is uploaded and ingested
  - answer is grounded in that document
  - document scope is set automatically
  - sources are shown in the response

### UC-03 Follow-Up Questions On Same Uploaded Document

- User uploads one document, then asks follow-up questions without re-uploading.
- Expected:
  - follow-up questions stay scoped to the same document
  - answers continue using the uploaded document
  - active scope is visible in the UI

### UC-04 Re-Select A Previously Uploaded Document

- User uploads multiple documents over time and later selects an older one from the sidebar.
- Expected:
  - selected upload becomes the active scope
  - follow-up answers use that selected document
  - switching scope does not require re-upload

### UC-05 Compare Multiple Documents

- User selects two or more remembered uploads and asks a comparison question.
- Expected:
  - response compares only the selected documents
  - similarities and differences are clearly separated
  - the model does not merge unrelated policies into one

### UC-06 Extract Text From Image Or Screenshot

- User uploads an image and asks for the text in the image.
- Expected:
  - chatbot returns extracted text directly
  - it should not behave like a normal QA response
  - no hallucinated summary should replace the raw text

### UC-07 Ask About Missing Information

- User asks a question whose answer is not in the uploaded document.
- Expected:
  - chatbot refuses cleanly
  - it should say the information is not found in the document
  - it should not fabricate an answer

### UC-08 Admin / Debug Use

- User checks health, models, stored docs, and reset behavior.
- Expected:
  - service status and document storage state are visible
  - reset clears vector DB and registry safely

## 4. Manual Test Cases

## A. UI And Session Flow

### TC-UI-01 Load Chat UI

- Steps:
  1. Start the backend.
  2. Open `/ui`.
- Expected:
  - chat UI loads successfully
  - model selector is visible
  - health badge is visible
  - chat input is usable

### TC-UI-02 Start New Chat

- Steps:
  1. Click `New Chat`.
- Expected:
  - a fresh chat session is created
  - old messages are not shown in the new session
  - previous chats remain in sidebar history

### TC-UI-03 Chat History Persistence

- Steps:
  1. Send 2-3 messages.
  2. Refresh the browser.
- Expected:
  - chats remain visible in the sidebar
  - previous messages reload from local storage

### TC-UI-04 Clear Active Document Scope

- Steps:
  1. Select or upload a document.
  2. Click `Clear Selection`.
- Expected:
  - active scope is removed
  - next message is no longer document-scoped

## B. Model Selection

### TC-MODEL-01 List Available Models

- Steps:
  1. Open `/models` or check the UI selector.
- Expected:
  - available installed/configured models are returned
  - one default model is selected

### TC-MODEL-02 Switch Model In UI

- Steps:
  1. Select a different model in the UI.
  2. Send a message.
- Expected:
  - the selected model is used for the response
  - selection persists on refresh

### TC-MODEL-03 Uninstalled Model Error

- Steps:
  1. Force a request using a model that is not installed.
- Expected:
  - a clear error is returned
  - the error explains that the model is not installed

### TC-MODEL-04 Memory Error Handling

- Steps:
  1. Request a model too large for the machine.
- Expected:
  - user-friendly memory error is returned
  - app does not crash silently

## C. Health And Admin Endpoints

### TC-API-01 Health Check

- Steps:
  1. Call `GET /health`.
- Expected:
  - response includes app status
  - Ollama connectivity is shown
  - document chunk count is shown

### TC-API-02 Root Endpoint

- Steps:
  1. Call `GET /`.
- Expected:
  - service metadata is returned
  - configured paths and model info are shown

### TC-API-03 Docs List

- Steps:
  1. Upload or ingest documents.
  2. Call `GET /docs-list`.
- Expected:
  - stored document metadata is returned
  - doc IDs and pages/chunks are visible

### TC-API-04 Reset DB

- Steps:
  1. Upload at least one file.
  2. Call `DELETE /reset-db`.
  3. Check `/docs-list`.
- Expected:
  - stored chunks are removed
  - document registry is cleared
  - app remains usable after reset

## D. Single-File Upload

### TC-UP-01 Upload Supported PDF

- Steps:
  1. Call `POST /upload` with a valid PDF.
- Expected:
  - request succeeds
  - chunk count is returned
  - a `doc_id` is returned

### TC-UP-02 Upload Unsupported File Type

- Steps:
  1. Call `POST /upload` with `.txt` or `.exe`.
- Expected:
  - request is rejected
  - status is `400`
  - unsupported extension is shown

### TC-UP-03 Upload Empty Or Unreadable File

- Steps:
  1. Upload a corrupted PDF or unreadable file.
- Expected:
  - request fails cleanly or returns extraction failure
  - app does not crash

### TC-UP-04 Chat-With-File Happy Path

- Steps:
  1. Call `POST /chat-with-file` with one valid document and one question.
- Expected:
  - file is ingested
  - answer is returned in one response
  - `active_source` and `active_doc_id` are returned

### TC-UP-05 Chat-With-File Unsupported Type

- Steps:
  1. Call `POST /chat-with-file` with an unsupported file type.
- Expected:
  - request is rejected with `400`

## E. Multi-File Upload

### TC-MULTI-01 Upload Multiple Supported Files

- Steps:
  1. Call `POST /upload-multiple` with 2-3 valid files.
- Expected:
  - each file gets a result entry
  - success/failed status is shown per file
  - total chunks are returned

### TC-MULTI-02 Mixed Valid And Invalid Files

- Steps:
  1. Upload valid PDFs plus one unsupported file.
- Expected:
  - valid files succeed
  - invalid file is skipped/rejected
  - batch result remains usable

### TC-MULTI-03 Multi-Upload UI Behavior

- Steps:
  1. Attach multiple files in the UI and send.
- Expected:
  - files upload successfully
  - sidebar refreshes with remembered uploads
  - system tells the user to select documents for comparison or follow-up

## F. Remembered Uploads And Registry

### TC-REG-01 Uploaded Docs List

- Steps:
  1. Upload one or more files.
  2. Call `GET /uploaded-docs`.
- Expected:
  - uploaded docs are listed
  - metadata includes doc ID, page count, chunk count, timestamps

### TC-REG-02 Delete Uploaded Document

- Steps:
  1. Upload a file.
  2. Call `DELETE /upload-document` using `doc_id` or `source`.
- Expected:
  - vector chunks are removed
  - registry entry is removed
  - local uploaded file is deleted when applicable

### TC-REG-03 Delete Missing Upload

- Steps:
  1. Call `DELETE /upload-document` for a nonexistent file/doc_id.
- Expected:
  - `404` is returned
  - app does not crash

### TC-REG-04 Re-Upload Same File

- Steps:
  1. Upload a file.
  2. Upload the same file again.
- Expected:
  - old scoped chunks are replaced instead of duplicated
  - answers remain stable

## G. Document-Scoped Chat

### TC-SCOPE-01 Ask Exact Lookup Question

- Steps:
  1. Upload a PDF with known facts.
  2. Ask a specific lookup question.
- Expected:
  - answer comes from the uploaded document
  - cited source/page is shown

### TC-SCOPE-02 Ask Summary Question

- Steps:
  1. Upload a document.
  2. Ask for a summary.
- Expected:
  - full-document context is used where needed
  - answer summarizes key points from the file

### TC-SCOPE-03 Ask Follow-Up Question

- Steps:
  1. Upload a file and ask one question.
  2. Ask a second question referring to “this document”.
- Expected:
  - chatbot keeps the active uploaded scope
  - second answer stays grounded in that file

### TC-SCOPE-04 Ask Not-Found Question

- Steps:
  1. Upload a file.
  2. Ask about information not present in the file.
- Expected:
  - chatbot says it cannot find the information in the uploaded document
  - no invented answer is produced

### TC-SCOPE-05 Smalltalk Clears Scope Influence

- Steps:
  1. Set an active document.
  2. Ask “hello” or “thanks”.
- Expected:
  - chatbot replies naturally
  - document scope should not force awkward document-grounded responses

## H. OCR And Direct Extraction

### TC-OCR-01 Screenshot Text Extraction

- Steps:
  1. Upload an image with clear text.
  2. Ask “extract the text” or “what does this image say”.
- Expected:
  - extracted text is returned directly
  - response is not just a summary

### TC-OCR-02 Scanned PDF Text Extraction

- Steps:
  1. Upload a scanned PDF.
  2. Ask for the text or a simple fact from it.
- Expected:
  - OCR/VLM fallback is used as needed
  - useful text is extracted and answerable

### TC-OCR-03 Image With No Readable Text

- Steps:
  1. Upload a blank or unreadable image.
  2. Ask for extracted text.
- Expected:
  - system returns a clean failure/not-found style response
  - no fabricated text

## I. Bundled PDF Ambiguity Handling

### TC-BUNDLE-01 Ambiguous Question On Bundled PDF

- Steps:
  1. Upload a bundled PDF containing multiple sub-guidelines.
  2. Ask a vague clause/support question without naming the sub-guideline.
- Expected:
  - chatbot asks for clarification
  - it should not answer from the wrong sub-guideline

### TC-BUNDLE-02 Named Sub-Guideline Narrowing

- Steps:
  1. Use the same bundled PDF.
  2. Ask the same question but name the sub-guideline explicitly.
- Expected:
  - context narrows to the named sub-guideline
  - answer comes from the correct section only

### TC-BUNDLE-03 Continuation Pages

- Steps:
  1. Ask about a section that continues across pages.
- Expected:
  - answer includes evidence from continuation pages
  - section identity is preserved across pages

## J. Multi-Document Comparison

### TC-COMP-01 Compare Two Documents In UI

- Steps:
  1. Upload two documents.
  2. Select both in the sidebar.
  3. Ask “compare their support scope”.
- Expected:
  - answer is comparison-oriented
  - each point is assigned to the correct document

### TC-COMP-02 Remove One Document From Comparison

- Steps:
  1. Select two documents.
  2. Deselect one.
  3. Ask a follow-up question.
- Expected:
  - chat returns to single-document scope
  - answer no longer compares both

### TC-COMP-03 Compare Unrelated Docs

- Steps:
  1. Select two unrelated documents.
  2. Ask a comparison question.
- Expected:
  - chatbot stays grounded
  - it should note lack of comparable evidence when needed

## K. OpenAI-Compatible API

### TC-OAI-01 List Models

- Steps:
  1. Call `GET /v1/models`.
- Expected:
  - OpenAI-style model list is returned

### TC-OAI-02 Chat Completion

- Steps:
  1. Call `POST /v1/chat/completions` with standard OpenAI-format messages.
- Expected:
  - valid OpenAI-style completion response is returned
  - assistant content contains the answer

### TC-OAI-03 System Prompt Handling

- Steps:
  1. Send a system prompt plus user question to `/v1/chat/completions`.
- Expected:
  - system prompt is respected in the final answer

## L. Optional Web Search

Only run these if `TAVILY_API_KEY` is configured and `tavily-python` is installed.

### TC-WEB-01 Current-Info Query

- Steps:
  1. Ask a current-event or latest-info question with no active document scope.
- Expected:
  - web search results are added when needed
  - answer uses web information instead of hallucinating

### TC-WEB-02 Scoped Document Query Should Not Trigger Web Search

- Steps:
  1. Select an uploaded document.
  2. Ask a document question.
- Expected:
  - chatbot stays within the document scope
  - web search should not override scoped document QA

### TC-WEB-03 Web Search Dependency Missing

- Steps:
  1. Leave web search unconfigured.
  2. Ask a latest/current-info question.
- Expected:
  - app still responds
  - no crash occurs because Tavily is absent

## M. Error Handling And Resilience

### TC-ERR-01 Ollama Offline

- Steps:
  1. Stop Ollama.
  2. Send a chat request.
- Expected:
  - health shows disconnected
  - request fails with a clear connection error

### TC-ERR-02 Timeout Handling

- Steps:
  1. Force a slow model or long-running request.
- Expected:
  - retry/timeout handling works
  - user sees a clean error if the request finally fails

### TC-ERR-03 Large File Upload

- Steps:
  1. Upload a large PDF.
- Expected:
  - app remains responsive
  - success or failure is explicit
  - no silent partial state

### TC-ERR-04 Repeated Reset And Re-Upload

- Steps:
  1. Upload docs.
  2. Reset DB.
  3. Upload again.
- Expected:
  - app recovers cleanly
  - no stale readonly DB errors

## 5. Suggested Acceptance Checklist

Use this as a high-level signoff list:

- UI loads and chat works
- model selector works
- health endpoint works
- single upload works
- multi-upload works
- remembered uploads work
- scoped follow-up chat works
- OCR/direct extraction works
- bundled PDF clarification works
- multi-document comparison works
- not-found refusal works
- reset and document deletion work
- OpenAI-compatible endpoints work
- optional web search path works when configured

## 6. Good Test Prompts

Use prompts like:

- `Summarize this document`
- `What is the application condition in this file?`
- `Extract the text from this image`
- `What does this section mean?`
- `Compare these two documents`
- `Is there any rule about late fees in this document?`
- `Tell me the support target`
- `Tell me the support target for ProjectLab`
- `What is the latest ...`  (only for optional web-search testing)

