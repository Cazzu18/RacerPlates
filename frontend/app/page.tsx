"use client";

import {useState} from "react";
import { predict } from "./(lib)/api";
import Image from "next/image";

export default function Home() {
  const[name, setName] = useState("Cheese Pizza");
  const [result, setResult] = useState<any>(null);


  return(
    <main className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">
        RacerPlates Demo
      </h1>

      <input className="border p-2 w-full mb-2" value={name} onChange={e=>setName(e.target.value)} />

      <button className="px-4 py-2 bg-black text-white rounded">
        Predict
      </button>

      {result && (
        <div className="mt-4">
          <div>Label: <b>{result.label}</b></div>
          {result.proba && <pre className="text-sm">{JSON.stringify(result.proba, null, 2)}</pre>}
        </div>
      )}
    </main>
  );
}