import os
import sys
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
from os.path import join, isfile
from tqdm import tqdm
import shutil

class PIEExtractor:
    def __init__(self):
        # 路径配置（适配 WSL）
        self._clips_path = '/mnt/f/Dataset_lr'
        self._annotation_path = '/mnt/e/PIE_dataset/annotations'
        self._pie_path = '/mnt/e/PIE_dataset'

    def get_annotated_frame_numbers(self, set_id):
        """
        Generates and returns a dictionary of videos and annotated frames for each video in the give set
        :param set_id: Set to generate annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<annotated_frame_id_0>,... <annotated_frame_id_n>]}
        """

        print("Generating annotated frame numbers for", set_id)
        annotated_frames_file = join(self._pie_path, "annotations", set_id, set_id + '_annotated_frames.csv')
        # If the file exists, load from the file
        if isfile(annotated_frames_file):
            with open(annotated_frames_file, 'rt') as f:
                annotated_frames = {x.split(',')[0]:
                                        [int(fr) for fr in x.split(',')[1:]] for x in f.readlines()}
            return annotated_frames
        else:
            # Generate annotated frame ids for each video
            annotated_frames = {v.split('_annt.xml')[0]: [] for v in sorted(os.listdir(join(self._annotation_path,
                                                                                         set_id))) if
                                v.endswith("annt.xml")}
            for vid, annot_frames in sorted(annotated_frames.items()):
                _frames = []
                path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
                tree = ET.parse(path_to_file)
                tracks = tree.findall('./track')
                for t in tracks:
                    if t.get('label') != 'pedestrian':
                        continue
                    boxes = t.findall('./box')
                    for b in boxes:
                        # Exclude the annotations that are outside of the frame
                        if int(b.get('outside')) == 1:
                            continue
                        _frames.append(int(b.get('frame')))
                _frames = sorted(list(set(_frames)))
                annot_frames.append(len(_frames))
                annot_frames.extend(_frames)

            with open(annotated_frames_file, 'wt') as fid:
                for vid, annot_frames in sorted(annotated_frames.items()):
                    fid.write(vid)
                    for fr in annot_frames:
                        fid.write("," + str(fr))
                    fid.write('\n')

        return annotated_frames

    def extract_and_save_images(self, sets_to_process=None):
        """
        Extracts images from clips and saves on hard drive
        :param extract_frame_type: Whether to extract 'all' frames or only the ones that are 'annotated'
                             Note: extracting 'all' frames requires approx. 3TB space whereas
                                   'annotated' requires approx. 1TB
        """
        set_folders = [f for f in sorted(os.listdir(self._clips_path))]
        if sets_to_process is not None:
            set_folders = [s for s in set_folders if s in sets_to_process]
        for set_id in tqdm(set_folders, desc="Extracting frames by set", ncols=80, file=sys.stdout):
            print(f"\nExtracting frames from {set_id}")
            set_folder_path = join(self._clips_path, set_id)
            extract_frames = self.get_annotated_frame_numbers(set_id)
            set_images_path = join(self._pie_path, "images", set_id)

            for vid, frames in tqdm(sorted(extract_frames.items()),
                                    desc=f"{set_id} videos", ncols=80, file=sys.stdout, leave=True):
                if vid != 'video_0012':
                    continue
                video_images_path = join(set_images_path, vid)
                num_frames = frames[0]
                frames_list = frames[1:]
                os.makedirs(video_images_path, exist_ok=True)

                vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.mp4'))
                success, image = vidcap.read()
                if not success:
                    tqdm.write(f"⚠️ Failed to open video {vid}")
                    continue

                with tqdm(total=num_frames, desc=f"{vid}", leave=False, ncols=80, file=sys.stdout) as pbar:
                    frame_num = 0
                    img_count = 0
                    while success:
                        if frame_num in frames_list:
                            img_count += 1
                            # ✅ 改为 .jpg 格式
                            img_path = join(video_images_path, "%05.f.jpg") % frame_num
                            if not os.path.isfile(img_path):
                                # 不存在, 保存
                                cv2.imwrite(img_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                print(f"Saved new {img_path}")
                            else:
                                # 存在, 验证是否损坏
                                try:
                                    with Image.open(img_path) as pil_img:
                                        pil_img.verify()  # 快速校验 (不加载全像素)
                                    print(f"Verified existing {img_path}")
                                except Exception as e:
                                    print(f"[WARN] {img_path} exists but damaged ({e}), re-saving")
                                    cv2.imwrite(img_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            pbar.update(1)
                        success, image = vidcap.read()
                        frame_num += 1

                if num_frames != img_count:
                    tqdm.write(f"{vid}: num images don't match {num_frames}/{img_count}")

if __name__ == "__main__":
    extractor = PIEExtractor()
    # ✅ 可指定只提取部分 set，例如：
    extractor.extract_and_save_images(sets_to_process=['set03'])
    # extractor.extract_and_save_images()
