import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    # looping through files in directory
    for filename in os.listdir(directory):
        #skips over files that are not html files
        if not filename.endswith(".html"):
            continue
        # opening a file and taking the href tag value and storing it as a variable in an array of pages
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    #corpus is total web pages
    #page is the index for corpus
    #damping_factor is used for probability of choosing a certain page

    #total number of pages
    N = len(corpus)

    #probability distribution
    distrib = {}
    
    #for page in the corpus
    for p in corpus:
        # pagerank default values for pages with/without links
        pr = (1 - damping_factor) / N
        # if there exists a page and the link is in the page
        if(len(corpus[p]) and p in corpus[page]):
            #divide number of total links by the dp factor
            pr += (damping_factor/ len(corpus[page]))
        # else divide total number of pages by total number of pages
        else:
            #equally distributes prob
            pr += (damping_factor/N)
        #map prob values to distribution
        distrib[p] = pr
    return distrib


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #assigns and initializes default values for pagerank values
    pRanks = {p: 0 for p in corpus}
    # inital sample through random choice of the corpus
    initPage = random.choice(list(corpus.keys()))
    for z in range(n):
        pRanks[initPage] += 1
        sample = transition_model(corpus, initPage, damping_factor)
        initPage = random.choice(list(sample.keys()))
       
    for p in pRanks:
        pRanks[p] = pRanks[p]/n
    return pRanks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    
    #gets previous ranked values
    prevRanks = {}
    #loops over corpus and assigns each page an equal value
    for p in corpus:
        prevRanks[p] = 1/N
    
    #create a new version of ranks, nteratively assigning ranks
    while True:
        #create new dict of ranks
        currRanks = {}

        #with the new currRanks dict, calculate the new ranks based on previous knowledge
        for page in corpus:
            #equal likely page
            currPR = (1 - damping_factor)/N
            # for a current page and its links
            for currPage, links in corpus.items():
                # if there exists links, the page is not the same as the original, and the page is apart of links
                if(links):
                    if(currPage != page):
                        if(page in links):
                            #calculate based on the dampening factor, pagerank from previous, and divide by the len of the links
                            currPR += damping_factor * (prevRanks[currPage]/len(corpus[currPage]))
                else:
                    # if no links then calculate based on dampening factor with equal distribution
                    currPR += damping_factor * (prevRanks[currPage]/N)
                #assign calculations to most recent version
                currRanks[page] = currPR
        #check for convergence with values closest to .001, return newest iteration

        for page in currRanks:
            #if the rounded value to the hundreds is greater then 0, restart cycle
            if(round(currRanks[page] - prevRanks[page], 3) > 0):
                prevRanks = currRanks.copy()
                break
        # for loop else statement which runs if completes loop of pages
        else:
            return currRanks


if __name__ == "__main__":
    main()
