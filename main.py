'''Code for the program to perform the analysis and crime category prediction
Author: Prasanth Varikallu'''
#Packages needed
from tkinter import *
import pandas as pd,glob,math,random
from haversine import haversine
import geocoder
from string import capwords
import matplotlib.pyplot as plt
import numpy as np,seaborn as sns

#Fields to display in the GUI
fields = ('Zipcode', 'Hour of the day')

txt =""


#A dictionary holding the police district and their corresponding zip codes.
DistLatLon = {1:'19145',2:'19149',3:'19147',5:'19128',6:'19107',7:'19115',
              8:'19154',9:'19130',12:'19142',14:'19144',15:'19149',16:'19104',
              17:'19146',18:'19143',19:'19151',22:'19121',24:'19124',25:'19124',
              26:'19125',35:'19141',39:'19140',77:'19153'}




'''The below steps are used to create a pandas data frame from multiple csv files 
and parse date time columns in the created data frame.''' 
frame = pd.DataFrame()
list_ = []
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
allFiles = glob.glob("./Incidents/*.csv")
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0, parse_dates=['DISPATCH_DATE'], date_parser=dateparse)
    list_.append(df)
frame = pd.concat(list_)


'''In the below code, we operate on the data by selecting only the necessary columns
and also processing the date and time parts of the code'''

frame=frame[["DC_DIST","DISPATCH_TIME","TEXT_GENERAL_CODE","DISPATCH_DATE"]]
frame['DISPATCH_TIME'] = frame['DISPATCH_TIME'].str.extract('(\d+):\d+:\d+')
frame['DISPATCH_TIME'] = frame['DISPATCH_TIME'].astype(int)
frame['WEEKDAY'] = frame['DISPATCH_DATE'].dt.dayofweek



'''The function used below is used to create a bar graph. 
It was taken from a script in kaggle.com'''
def plot_bar(df, title, filename):
    grids = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
    sns.set_style(random.choice(grids))#Setting the style for the plot display window
    sns.despine()
    p = (
        'Set2', 'Paired', 'colorblind', 'husl', #This set is used for the random color generation in the bar plots
        'Set1', 'coolwarm', 'RdYlGn', 'spectral'
    )
    
    vals = ['bar','barh']
    bar = df.plot(kind=random.choice(vals),#Setting the bar plot characteristics
                  title=title,
                  fontsize=8,
                  figsize=(12,8),
                  stacked=False,
                  width=1,
                  color = sns.color_palette(np.random.choice(p), len(df)),
    )

    bar.figure.savefig(filename)#Save the figure locally.
    plt.show()#Shows the bar plot in a window.



#The below function is used to set up the top crime data to be plotted into a graph.
def plot_top_crimes(df, column, title, fname, items=0):
    by_col         = df.groupby(column) 
    col_freq       = by_col.size()#Creates a frequency count data format.
    col_freq.index = col_freq.index.map(capwords)
    col_freq.sort(ascending=True, inplace=True)#Sorting the column.
    plot_bar(col_freq[slice(-1, - items, -1)], title, fname)#Calling the plot_bar function



'''The below functions do the same thing as the above function, 
except that they give different statistics.
1. weekday_stats will give weekly crime distribution.
2. district_wise will give district wise counts of the crime categories
3. time_of_day will give hourly distribution of the categories.'''

def weekday_stats(df):
    by_col = df.groupby("WEEKDAY")
    col_freq = by_col.size()
    col_freq.sort(ascending=True, inplace=True)
    plot_bar(col_freq, "Daywise Counts", "weekday_stats.png")
def district_wise(df):
    by_col = df.groupby("DC_DIST")
    col_freq = by_col.size()
    col_freq.sort(ascending=True, inplace=True)
    plot_bar(col_freq, "Districtwise Counts", "district_stats.png")
def time_of_day(df):
    by_col = df.groupby("DISPATCH_TIME")
    col_freq = by_col.size()
    plot_bar(col_freq, "Crime Counts based on time of day", "timeofday_stats.png")
def total_counts(df):
    by_col = df.groupby("TEXT_GENERAL_CODE")
    col_freq = by_col.size()
    col_freq.sort(ascending=True, inplace=True)
    plot_bar(col_freq, "Total Crime Counts", "Crime_stats.png")




#The below function gives the frequency counts of the crimes given a time and zipcode
def freq_counts(entries,frame):
    district = 0
    min_dis = 25.0
    time = int(entries['Hour of the day'].get()) #The .get() method is used to fetch the values from the GUI text box
    myzip = entries['Zipcode'].get()
    zipcd = geocoder.google(myzip) #Converts the entered zip code to latitudes and longitudes.
    for key in DistLatLon:
        if min_dis == 0.0:
            break
        i = DistLatLon[key]
        pzip = geocoder.google(i)
        if (pzip.latlng):
            dis = haversine(zipcd.latlng,pzip.latlng,miles = True)#Haversine function calculates the distance 
            #between two locations given their latitudes and longitudes.
            if dis < min_dis:
                min_dis = dis
                district = key#Identifying which district the given zipcode falls under.
        
        else: #The else block des the same thing as the if block. This is due a bug in geocoder library. 
        #Sometimes, it doesn't fetch the latitudes and lonitudes as expected.
            pzip = geocoder.google(i)
            dis = haversine(zipcd.latlng,pzip.latlng,miles = True)
            if dis < min_dis:
                min_dis = dis
                district = key
    #Filtering the data frames with the district and time given.
    frame= frame[(frame["DC_DIST"]==district) & (frame["DISPATCH_TIME"]==time)]
    frame=frame[["DC_DIST","DISPATCH_TIME","TEXT_GENERAL_CODE"]]
    #Calling the plotting function to set it ready for plotting.
    plot_top_crimes(frame, 'TEXT_GENERAL_CODE','Top Crime Categories','category.png')

#The below function applies the naive bayes classifier to the training data and predict the results. 
def naive_bayes(entries,frame):
    txt = ""
    district = 0
    min_dis = 25.0
    docsCount = len(frame) #Counting the total number of rows.
    #Creating the vocabulary.
    vocabulary = pd.unique(frame["DC_DIST"]).tolist() 
    time_list = pd.unique(frame["DISPATCH_TIME"]).tolist()
    vocabulary.extend(time_list)
    vocabulary = sorted(set(vocabulary))
    v_len = len(vocabulary)#Calculating the length of library.
    #The .get() method is used to fetch the values from the GUI text box
    time = int(entries['Hour of the day'].get()) 
    myzip = entries['Zipcode'].get()
    
    #Calculating the number of documents in each class
    by_col         = frame.groupby("TEXT_GENERAL_CODE") 
    col_freq       = by_col.size().to_dict()
    
    
    def COUNTTOKENSOFTERM(df, t): #Function to calculate number of occurances of a term in the documents related to the class
        cnt = 0
        cnt += len(df[df['DC_DIST']==t])
        cnt += len(df[df['DISPATCH_TIME']==t])
        return cnt
    
    #Getting the classes as a list.
    classes = list(col_freq.keys())
    prior = {}#Dictionary to calculate prior class probability
    condprob = []#List to hold conditional probility.
    for cls in classes:
        NinC = col_freq[cls]
        prior[cls] = NinC/docsCount
        denominator = col_freq[cls] * 2 + v_len
        df = frame[frame['TEXT_GENERAL_CODE'] == cls]
        for t in vocabulary:
            Tct = COUNTTOKENSOFTERM(df[["DC_DIST","DISPATCH_TIME"]], t) #count of all occurances of a term in the class.
            condprob.append((t,cls,(Tct+1)/denominator)) #Tct+1 indicates the laplace smoothing
    #Converting zipcode to latitides and logitudes.
    #The if and else staments do the same thing. They identify the police district for the given zipcode.
    zipcd = geocoder.google(myzip)
    for key in DistLatLon:
        if min_dis == 0.0:
            break
        i = DistLatLon[key]
        pzip = geocoder.google(i)
        if (pzip.latlng):
            dis = haversine(zipcd.latlng,pzip.latlng,miles = True)
            if dis < min_dis:
                min_dis = dis
                district = key
        else:
            pzip = geocoder.google(i)
            dis = haversine(zipcd.latlng,pzip.latlng,miles = True)
            if dis < min_dis:
                min_dis = dis
                district = key
    txt+="The entered zipcode falls under police district number: "+str(district)+"\n\n"
    
    
    doc_rd = [district,time]
    best_class_scores = {}
    for c in prior: #loop to calculate for each class
            best_class_scores[c] = math.log(prior[c])
            for w in doc_rd: #loop to iterate through all words in the document and claculate the score
                for tup in condprob:
                    if tup[0] == w and tup[1] == c: #matching the tuple and class to get the probability value
                        best_class_scores[c] += math.log(tup[2]) #log calculations are used to avoid floating point underflow
    
        
    txt+="For the given location and time:\n\n"
    for c in best_class_scores:
        t1 = "Crime Category: "+c
        t2 = "Score: "+str('{0:.2f}'.format(best_class_scores[c]))
        txt+=t1.ljust(60,'-') + t2+"\n"   
    txt+="\n\nBEST MATCHED CATEGORY: "+str(max(best_class_scores, key=best_class_scores.get))+"\n\n"
    
    msg.config(text = txt) #This function puts the output in the GUI window.
    
    
#This fuction is used to create the GUI for the project.       
def makeform(root, fields):
   entries = {}
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=25, text=field+": ", anchor='w')
      ent = Entry(row)
      ent.insert(0,"0")
      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries[field] = ent
   return entries

#This is the main program.
if __name__ == '__main__':
   root = Tk()
   root.state('zoomed')
   ents = makeform(root, fields)
   #The buttons are declared below.
   b = Button(root, text='Using Naive Bayes',
          command=(lambda e=ents: naive_bayes(e,frame)))
   b.pack()

   b = Button(root, text='Frequency Counts',
          command=(lambda e=ents: freq_counts(e,frame)))
   b.pack()
   
   b = Button(root, text='Weekday Counts',
          command=(lambda e=ents: weekday_stats(frame)))
   b.pack(side = LEFT)
   
   b = Button(root, text='Districtwise Counts',
          command=(lambda e=ents: district_wise(frame)))
   b.pack(side = LEFT)
   
   b = Button(root, text='Hourly Counts',
          command=(lambda e=ents: time_of_day(frame)))
   b.pack(side = LEFT)
   
   msg = Message(root,relief = SUNKEN)
   msg.pack(side = RIGHT,padx = 30)
   root.mainloop()
