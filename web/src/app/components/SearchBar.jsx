
import { Input } from "@nextui-org/react";
import { throttle } from "lodash";
import { useRef } from "react";


export function SearchBar({ search }) {

    const fileRef = useRef(null);

    const textOnChange = throttle((e) => {
        search({ text: e.target.value });
    }
    , 2000);

    
    return (
      <div className="relative mb-4 flex justify-center items-center">
        <Input
          type="search"
          placeholder="Search for images..."
          onChange={textOnChange}
        />
        <div className="absolute inset-y-0 right-5 flex items-center pr-3 cursor-pointer" onClick={() => {
            fileRef.current.click();
    
        }}>
          <svg
            id="Layer_1"
            style={{ enableBackground: "new 0 0 64 64;" }}
            version="1.1"
            viewBox="0 0 64 64"
            xmlSpace="preserve"
            width={50}
            fill="white"
            height={50}
            xmlns="http://www.w3.org/2000/svg"
            xmlnsXlink="http://www.w3.org/1999/xlink"
          >
            <g>
              <g id="Icon-Image" transform="translate(278.000000, 232.000000)">
                <path
                  class="st0"
                  d="M-226.2-181.6h-39.5c-2.3,0-4.2-1.9-4.2-4.2v-28.2c0-2.3,1.9-4.2,4.2-4.2h39.5     c2.3,0,4.2,1.9,4.2,4.2v28.2C-222-183.5-223.9-181.6-226.2-181.6L-226.2-181.6z M-265.8-215.5c-0.8,0-1.4,0.6-1.4,1.4v28.2     c0,0.8,0.6,1.4,1.4,1.4h39.5c0.8,0,1.4-0.6,1.4-1.4v-28.2c0-0.8-0.6-1.4-1.4-1.4H-265.8L-265.8-215.5z"
                  id="Fill-12"
                />
                <path
                  class="st0"
                  d="M-238.9-201.5c-3.1,0-5.5-2.5-5.5-5.5s2.5-5.5,5.5-5.5s5.5,2.5,5.5,5.5     S-235.9-201.5-238.9-201.5L-238.9-201.5z M-238.9-210c-1.6,0-2.9,1.3-2.9,2.9c0,1.6,1.3,2.9,2.9,2.9c1.6,0,2.9-1.3,2.9-2.9     C-236-208.7-237.3-210-238.9-210L-238.9-210z"
                  id="Fill-13"
                />
                <polyline
                  class="st0"
                  id="Fill-14"
                  points="-231.4,-182.1 -254.5,-203.8 -267.7,-191.6 -269.5,-193.5 -254.5,-207.4 -229.6,-184      -231.4,-182.1    "
                />
                <polyline
                  class="st0"
                  id="Fill-15"
                  points="-224.2,-189.3 -231.9,-195.5 -238.3,-190.2 -240,-192.3 -231.9,-198.9 -222.6,-191.3      -224.2,-189.3    "
                />
              </g>
            </g>
          </svg>
          <Input
          ref={fileRef}
          className="hidden"
            type="file"
            accept="image/*"
            onChange={(e) => 
              search({ files: e.target.files })
            }
          />
        </div>
      </div>

      //   <form
      //     onSubmit={(e) => {
      //       e.preventDefault();
      //       const formData = new FormData(e.target);
      //       const text = formData.get("text");
      //       search({text});
      //     }}
      //     className="relative mb-2 flex justify-center items-center"
      //   >
      //     <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
      //       <svg
      //         className="w-4 h-4 text-gray-500 dark:text-gray-400"
      //         aria-hidden="true"
      //         xmlns="http://www.w3.org/2000/svg"
      //         fill="none"
      //         viewBox="0 0 20 20"
      //       >
      //         <path
      //           stroke="currentColor"
      //           strokeLinecap="round"
      //           strokeLinejoin="round"
      //           strokeWidth="2"
      //           d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"
      //         />
      //       </svg>
      //     </div>

      //     <input
      //       type="search"
      //       name="text"
      //       id="default-search"
      //       className="block w-full p-4 pl-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
      //       placeholder="Search for images..."
      //       required
      //     />

      //     {/* File Input */}
      //     <div
      //       className=" inset-y-0 right-0 flex items-center pr-3 cursor-pointer"
      //       style={{ right: "6em" }}

      //     >
      //       <input
      //         type="file"
      //         name="file"
      //         id="file"
      //         title="Upload an image"

      //         // className="hidden"
      //         accept="image/*"
      //         onChange={(e) => {
      //             // const file = e.target.files[0];
      //             // const fileReader = new FileReader();
      //             // fileReader.onloadend = () => {
      //             //     const image = fileReader.result;
      //             //     search({image});
      //             // };

      //             // fileReader.readAsDataURL(file);
      //             search({ files: e.target.files });
      //         }}
      //       />
      //       <svg
      //         className="feather feather-image"
      //         fill="none"
      //         height="24"
      //         stroke="currentColor"
      //         strokeLinecap="round"
      //         strokeLinejoin="round"
      //         strokeWidth="2"
      //         viewBox="0 0 24 24"
      //         width="24"
      //         xmlns="http://www.w3.org/2000/svg"
      //       >
      //         <rect height="18" rx="2" ry="2" width="18" x="3" y="3" />
      //         <circle cx="8.5" cy="8.5" r="1.5" />
      //         <polyline points="21 15 16 10 5 21" />
      //       </svg>
      //     </div>

      //     <button
      //       type="submit"
      //       className="text-white absolute right-10 bottom-2.5 bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
      //     >
      //       Search
      //     </button>
      //   </form>
    );
}