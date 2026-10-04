from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

text= '''
'Here\'s a possible tweet about AI:\n\n"Artificial intelligence is revolutionizing the way we live and work. From personalized medicine to self-driving cars, AI is transforming industries and changing the world. What do you think is the most exciting AI development on the horizon? #AI #FutureOfTech"', 'linkedin': 'Here\'s a potential LinkedIn post about AI:\n\n**Title:** "The Future of Work: How AI is Revolutionizing Industries and Redefining Careers"\n\n**Post:**\n\n"As we continue to navigate the ever-changing landscape of the modern workplace, one thing is clear: Artificial Intelligence (AI) is no longer a futuristic concept, but a present-day reality that\'s transforming industries and redefining careers.\n\nFrom augmenting human capabilities to automating routine tasks, AI is having a profound impact on the way we work, collaborate, and innovate. Whether you\'re a seasoned professional or a recent graduate, understanding the role of AI in the job market is crucial to staying ahead of the curve.\n\nHere are just a few examples of how AI is revolutionizing industries:\n\n* **Healthcare:** AI is helping doctors diagnose diseases more accurately and efficiently, freeing up time for more critical and high-value tasks.\n* **Finance:** AI-powered systems are detecting and preventing financial crimes, streamlining transactions, and providing personalized investment advice.\n* **Customer Service:** AI-powered chatbots are revolutionizing the way companies interact with their customers, providing 24/7 support and improving customer satisfaction.\n\nAs AI continues to advance, we can expect to see even more exciting innovations in areas like:\n\n* **Cybersecurity:** AI-powered systems will be able to detect and prevent cyber threats in real-time, protecting our digital assets and ensuring business continuity.\n* **Education:** AI will enable personalized learning experiences, tailoring education to individual needs and abilities.\n* **Environmental Sustainability:** AI will help us better understand and mitigate the impact of human activities on the environment, driving more sustainable practices.\n\nBut what does this mean for us? As AI becomes more prevalent, we\'ll need to adapt and develop new skills to stay relevant in the job market. Here are a few takeaways:\n\n* **Upskill and Reskill:** Invest in courses and training programs that focus on AI, data science, and related technologies.\n* **Collaborate and Communicate:** Develop strong collaboration and communication skills to work effectively with AI systems and human colleagues.\n* **Emphasize Human Touch:** Focus on developing skills that are uniquely human, such as empathy, creativity, and critical thinking.\n\nAs AI continues to shape the future of work, one thing is certain: it\'s not about replacing humans, but about augmenting our abilities and creating new opportunities for growth and innovation.\n\nWhat are your thoughts on the future of work with AI? Share your perspectives and experiences in the comments below!\n\n**#AI #FutureOfWork
'''

loader= PyPDFLoader('dl-curriculum.pdf')

docs= loader.load()

splitter= CharacterTextSplitter(
    chunk_size=100,
    separator=' ',
    chunk_overlap=0
)

result= splitter.split_documents(docs)


print(result[0].page_content)
print(len(result))
