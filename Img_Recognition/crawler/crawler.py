# from icrawler.builtin import BingImageCrawler
# import os

# #we are building cats image detection that's why we put cat here
# #if you want some other images then put that name in classes list
# classes=['cars images'] 
# number=100
# #here root directory is find your root directory there u will find 
# #new file name data in which all images are saved.
# for c in classes:
#     bing_crawler=BingImageCrawler(storage={str(os.path.dirname(__file__) + "/images"):f'p/{c.replace(" ",".")}'})
#     bing_crawler.crawl(keyword=c,filters=None,max_num=number,offset=0)




import os
from icrawler.builtin import BingImageCrawler

# Define the directory to save images
image_save_dir = os.path.dirname(__file__) + "/images/p"
os.makedirs(image_save_dir, exist_ok=True)

# Initialize the BingImageCrawler
bing_crawler = BingImageCrawler(
    downloader_threads=4, # Number of threads for downloading
    storage={'root_dir': image_save_dir}
)

# Define the search keyword and maximum number of images
classes = ['cars']
max_images = 50

# Start crawling
for c in classes:
    print(f"Crawling {max_images} images for '{c}' from Bing...")
    bing_crawler.crawl(keyword=c, max_num=max_images)

print(f"Finished crawling. Images saved to: {image_save_dir}")