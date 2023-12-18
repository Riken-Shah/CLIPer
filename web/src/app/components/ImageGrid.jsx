// import Image from 'next/image'
import { blurHashToDataURL } from "../utils.js";
import React from "react";

import {
  Image,
  Popover,
  Input,
  PopoverContent,
  Button,
  PopoverTrigger,
} from "@nextui-org/react";

export function ImageGrid({ images, search }) {
  return (
    <div className="columns-2 gap-4 sm:columns-3 xl:columns-4 2xl:columns-5">
      {images &&
        images.map(
          ({
            id: photo_id,
            photo_url,
            thumbnail_url,
            //   photo_image_url,
            //   photo_aspect_ratio,
            //   photo_width,
            //   photo_height,
            //   blur_hash,
            //   photo_description,
            //   ai_description,
            similarity,
          }) => (
            <div className="group block mb-2 relative" key={photo_id}>
              <Image
                alt="Woman listing to music"
                className="object-cover mb-4"
                height={400}
                src={`http://localhost:1001/${thumbnail_url}`}
                width={400}
                loading="lazy"
                isBlurred
              />

              {/* // blured backgroun */}
              <div className="absolute top-2 right-2 z-10">
                <Button
                  isIconOnly
                  color="black"
                  className="bg-gradient-to-tr from-pink-500 to-yellow-500"
                  variant="faded"
                  ariaLabel="Take a photo"
                  onClick={(e) => {
                    e.preventDefault();
                    fetch(
                      `http://localhost:1001/${photo_url}?auto=format&fit=crop&w=480&q=80`
                    ).then((r) => {
                      // console.log(r.blob())
                      r.blob().then((blob) => search({ files: [blob] }));
                    });
                  }}
                >
                  <svg
                    height="512px"
                    id="Layer_1"
                    fill="white"
                    style={{ enableBackground: "new 0 0 512 512;" }}
                    version="1.1"
                    viewBox="0 0 512 512"
                    width="512px"
                    xmlSpace="preserve"
                    xmlns="http://www.w3.org/2000/svg"
                    xmlnsXlink="http://www.w3.org/1999/xlink"
                  >
                    <path d="M344.5,298c15-23.6,23.8-51.6,23.8-81.7c0-84.1-68.1-152.3-152.1-152.3C132.1,64,64,132.2,64,216.3  c0,84.1,68.1,152.3,152.1,152.3c30.5,0,58.9-9,82.7-24.4l6.9-4.8L414.3,448l33.7-34.3L339.5,305.1L344.5,298z M301.4,131.2  c22.7,22.7,35.2,52.9,35.2,85c0,32.1-12.5,62.3-35.2,85c-22.7,22.7-52.9,35.2-85,35.2c-32.1,0-62.3-12.5-85-35.2  c-22.7-22.7-35.2-52.9-35.2-85c0-32.1,12.5-62.3,35.2-85c22.7-22.7,52.9-35.2,85-35.2C248.5,96,278.7,108.5,301.4,131.2z" />
                  </svg>
                </Button>
              </div>

              <div className="absolute bottom-2 right-2 group-hover:visible invisible">
                <Popover placement="top" showArrow offset={10} classNames="">
                  <PopoverTrigger>
                    <Button className="bg-gradient-to-tr from-pink-500 to-yellow-500">
                      More Options
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[240px]">
                    {(titleProps) => (
                      <div className="px-1 py-2 w-full">
                        <div className="mt-2 flex flex-col gap-2 w-full">
                          <Button
                            // defaultValue="100%"
                            label="Width"
                            // size="sm"
                            variant="bordered"
                          >
                            {" "}
                            Download Image{" "}
                          </Button>
                          <Input
                            defaultValue={photo_id}
                            label="File ID"
                            size="sm"
                            disabled
                            variant="bordered"
                          />
                          <Input
                            defaultValue={photo_url}
                            label="File Path"
                            size="sm"
                            disabled
                            variant="bordered"
                          />
                          <Input
                            defaultValue="24px"
                            label="Height"
                            size="sm"
                            variant="bordered"
                          />
                          <Input
                            defaultValue="30px"
                            label="Max. height"
                            size="sm"
                            variant="bordered"
                          />
                        </div>
                      </div>
                    )}
                  </PopoverContent>
                </Popover>
              </div>
            </div>
          )
        )}
    </div>
  );
}
