# RacerPlates Frontend

Next.js 16 dashboard for browsing Winslow Dining menu items and requesting satisfaction predictions from the FastAPI backend.

## Prerequisites
- Node.js 18+
- Backend running locally (default `http://localhost:8000`) or a reachable deployment.

## Setup
```bash
cd frontend
npm install
```

Create `.env.local` with:
```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## Run
```bash
npm run dev
#open http://localhost:3000
```

## Build
```bash
npm run build
npm run start
```

## Notes
- API calls are defined in `app/(lib)/api.ts` and default to the fusion MLP; you can pass `model` to request `sbert_fusion_linear` or `oracle_knn_embeddings`.
- If you change backend routes, update the client constants accordingly.
