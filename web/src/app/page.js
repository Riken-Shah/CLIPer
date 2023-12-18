"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Modal } from "./components/Modal";
import { SearchBar } from "./components/SearchBar";
import { ImageGrid } from "./components/ImageGrid";
import axios from "axios";
import _ from "lodash";

const API_URL = "http://localhost:1001";

export default function Home() {
  // Application state
  const [result, setResult] = useState(null);

  function loadImage(selectedFile) {
    var reader = new FileReader();

    reader.onload = function (event) {
      let imgObject = new Object();
      imgObject.src = event.target.result;
    };

    reader.readAsDataURL(selectedFile);
  }

  const search = async ({
    text,
    files,
  }) => {
setResult(null);
    
    if (text) {
      const resp = await axios.post(
        API_URL + "/search",
        {
          text,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      setResult(resp.data);
    } else {
      const acceptedFiles = files;
      let data = new FormData();
      for (let i = 0; i < acceptedFiles.length; i++) {
        const file = acceptedFiles[i];
        data.append("fileToUpload[]", file);
        loadImage(file);
      }

      const resp = await axios.post(API_URL + "/search", data, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResult(resp.data);
    }
  };

  const throttledSearch = useCallback(
    _.throttle(search, 1000, { leading: false }),
    []
  );

  return (
    <main className="mx-auto max-w-[1960px] p-4 relative">
      {/* <Modal currentImage={currentImage} setCurrentImage={setCurrentImage} /> */}
      <SearchBar search={throttledSearch} />
      <ImageGrid images={result} search={throttledSearch} />
    </main>
  );
}
