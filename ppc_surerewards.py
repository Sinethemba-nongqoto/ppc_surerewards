"""

	Simple Streamlit webserver application for serving developed classification
	models.

	Author: Explore Data Science Academy.

	Note:
	---------------------------------------------------------------------
	Please follow the instructions provided within the README.md file
	located within this directory for guidance on how to use this script
	correctly.
	---------------------------------------------------------------------

	Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
#import joblib,os

# Data dependencies
import pandas as pd

# graph in streamlit
import altair as alt
## For KPIS 
from streamlit_metrics import metric, metric_row

# MySQL 
#Import labraries
import pymysql


import warnings
warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file



# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app

# Connecting To The Database Using MySQL
conn=pymysql.connect(host='us-mm-dca-d04d6c3c8e49.g5.cleardb.net',port=int(3306),user='b04c9045b8a037',passwd='f9f69807ebbcc01',db='heroku_c6143c8aee66786')


def main():

	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages


	# Loading Heading Images 

	## Function that automatically load images
	from PIL import Image
	
	def load_image(image_file):
		img = Image.open(image_file)
		return img
	
	col1, col2, col3 = st.columns([1,20,1])

	with col1:
		st.write("")

	with col2:
		st.image(load_image("resources/imgs/Heading_Logo.png"),width=750)
	with col3:
		st.write("")

	def line_graph(source, x, y):
		# Create a selection that chooses the nearest point & selects based on x-value
		hover = alt.selection_single(fields=[x],nearest=True,on="mouseover",empty="none",	)

		lines = (alt.Chart(source).mark_line(point="transparent").encode(x=x, y=y).transform_calculate(color='datum.delta < 0 ? "red" : "green"'))

		# Draw points on the line, highlight based on selection, color based on delta
		points = (lines.transform_filter(hover).mark_circle(size=65).encode(color=alt.Color("color:N", scale=None)))

		# Draw an invisible rule at the location of the selection
		tooltips = (alt.Chart(source).mark_rule(opacity=0).encode(x=x,y=y,tooltip=[x, y, alt.Tooltip("delta", format=".2%")],).add_selection(hover))
		return (lines + points + tooltips).interactive()
	
	
	
	st.markdown("<h1 style='text-align: center; color: red;'>PPC130 Surerewards Insights</h1>", unsafe_allow_html=True)



	## Writing the latest time update
	from datetime import datetime
	import pytz

	SA_time = pytz.timezone('Africa/Johannesburg') 
	datetime_SA = datetime.now(SA_time)

	metric("Latest Time Update", datetime_SA.strftime('%Y-%m-%d %H:%M %p'))

	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	options = ["Prediction","Prediction Page"]
	selection = st.sidebar.selectbox("Select Page", options)





	# Building out the "Information" page
	if selection == "Prediction Page":
		
		st.markdown("<h3 style='text-align: center; color: black;'>This Page focuses on predicting number of bag and setting dynamic daily targets</h3>", unsafe_allow_html=True)




		num_bags =pd.read_sql_query("select cast(r.createdAt as date) as date,sum(ppc_surebuild + ppc_surecast + ppc_surecem + ppc_suretech + ppc_surewall) as number_of_bags from receipts as r inner join receiptdata as rdata on r.id =rdata.receipt_id where r.status in ('approved','Limit reached.') and cast(r.createdAt as date)  between '2022-02-15' and current_date() group by date" ,conn)
		content = {'date': list(pd.to_datetime(num_bags['date'], errors='coerce')),'number_of_bags': list(num_bags['number_of_bags'])}
		df = pd.DataFrame(content).set_index('date')

		from statsmodels.tsa.seasonal import seasonal_decompose

		st.info("The number of bags by date is found to constist of two hidden properties,after decomposition the first proterty is a trend that shows that the number of bags bought by surerewards is increase with time.Second properties is that the number of bags has  a weekly seasonality,that is it varies depending on the day of the week(with the weekend having the lowest sales (dip ))")


		if st.checkbox('Show trend'): # data is hidden if box is unchecked
			ax = seasonal_decompose(df['number_of_bags'])

			fig =ax.plot()


			fig.set_size_inches((12, 9))
			# Tight layout to realign things
			fig.tight_layout()
			#plt.show()

			#results.plot();
			st.pyplot(fig)
		st.markdown("<h5 style='text-align: center; color: black;'>How the model works</h5>", unsafe_allow_html=True)
		if st.checkbox('Show How the works'):
		
			st.info("The model trained is a Recurrent Neural Network model,  LSTM ( Long Short Term Memory) . This model is a Time Series Model that use historical data for forecasting/ predicting the number of bags in the upcoming week")			

	# Building out the predication page
	if selection == "Prediction":
		st.markdown("<h1 style='text-align: center; color: red;'>Prediction Page</h1>", unsafe_allow_html=True)
		st.markdown("<h3 style='text-align: center; color: black;'>This Page focuses on predicting number of bag and setting dynamic daily targets</h3>", unsafe_allow_html=True)
		
	

		# Loaading Datasets
		num_bags =pd.read_sql_query("SELECT cast(r.createdAt as date) as date, sum(ppc_surebuild + ppc_surecast + ppc_surecem + ppc_suretech + ppc_surewall) as number_of_bags from users as u inner join receiptdata as r on u.id =r.agent_id where action = 'approved' and cast(r.createdAt as date) >= '2022-02-15'  group by Date",conn)
		num_reg =pd.read_sql_query(" Select count(*) as num_of_reg,cast(createdAt as date) as date  from users where cast(createdAt as date) >= '2022-02-15'  group by date order by date",conn)
		num_promo_reg = pd.read_sql_query("Select cast(createdAt as date) as date,count(*) as No_Promocode from users where code = 'PPC130' and cast(createdAt as date)>='2022-02-15'  group by date order by Date",conn)
		num_receipts=pd.read_sql_query("SELECT count(*) as no_of_receipts_upload ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Approved','Limit reached.') and cast(receipts.updatedAt as date) >='2022-02-15'  group by date",conn)
		num_valid_receipts=pd.read_sql_query("SELECT count(*) as no_of_valid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  Not in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Unprocessed') and cast(receipts.updatedAt as date) >= '2022-02-15'  group by date",conn)
		num_invalid_receipts=pd.read_sql_query("SELECT count(*) as no_of_invalid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible') and cast(receipts.updatedAt as date) >= '2022-02-15'  group by date",conn)

		num_user_r_upload=pd.read_sql_query("SELECT count(distinct(users.id)) as no_of_receipts_upload ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Approved','Limit reached.') and cast(receipts.updatedAt as date) >= '2022-02-15'  group by date",conn)
		
		num_user_valid_receipts=pd.read_sql_query("SELECT count(distinct(users.id)) as no_of_users_valid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  Not in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Unprocessed') and cast(receipts.updatedAt as date) >= '2022-02-15'  group by date",conn)
		num_user_invalid_receipts=pd.read_sql_query("SELECT count(distinct(users.id))as no_of_users_invalid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible') and cast(receipts.updatedAt as date) >= '2022-02-15'  group by date",conn)
		
		num_bags["delta"] = (num_bags["number_of_bags"].pct_change()).fillna(0)

		st.altair_chart(line_graph(num_bags,"date","number_of_bags"), use_container_width=True)
		
		## Grouping By Weekday
		num_bags['date'] = pd.to_datetime(num_bags['date'], errors='coerce')
		num_bags['weekday'] = num_bags["date"].dt.day_name()


			

		num_reg["delta"] = (num_reg["num_of_reg"].pct_change()).fillna(0)
		st.altair_chart(line_graph(num_reg ,"date","num_of_reg").interactive(), use_container_width=True)

		num_receipts["delta"] = (num_receipts["no_of_receipts_upload"].pct_change()).fillna(0)
		st.altair_chart(line_graph(num_receipts ,"date","no_of_receipts_upload").interactive(), use_container_width=True)

		mean_weekday =num_bags.groupby(['weekday']).mean()
		sum_weekday =num_bags.groupby(['weekday']).sum()

		sorted_weekdays = ['Sunday','Saturday','Friday','Thursday','Wednesday','Tuesday','Monday']

		sort_mean_week_dct={}
		sort_sum_week_dct={}
		for i in sorted_weekdays:
			sort_mean_week_dct[i]=round(mean_weekday['number_of_bags'][i])
			sort_sum_week_dct[i]=round(sum_weekday['number_of_bags'][i])

		plot_week=pd.DataFrame(index=sorted_weekdays)
		plot_week['Average']=sort_mean_week_dct.values()
		plot_week['Total']=sort_sum_week_dct.values()




		# Plot double bars

		fig, ax = plt.subplots(figsize=(10,5))
		plt.tight_layout()

		y = np.arange(len(sorted_weekdays))  # Label locations
		width = 0.4

		ax.barh(y + width/2, plot_week['Average'], width, label='Average')
		ax.barh(y - width/2, plot_week['Total'], width, label='Total')

		# Format ticks
		ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

		# Create labels
		rects = ax.patches
		for rect in rects:
			
    		# Get X and Y placement of label from rect.
			x_value = rect.get_width()
			y_value = rect.get_y() + rect.get_height() / 2
			space = 5
			ha = 'left'
			if x_value < 0:
				space *= -1
				ha = 'right'
			label = '{:.0f}'.format(x_value)
			plt.annotate(label,(x_value, y_value), xytext=(space, 0),textcoords='offset points',va='center',ha=ha)

		# Set y-labels and legend
		ax.set_yticklabels(sorted_weekdays)
		ax.legend(loc='lower right')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		# To show each y-label, not just even ones
		plt.yticks(np.arange(min(y), max(y)+1, 1.0))
		st.pyplot(fig)




		## To be Delete

		#st.bar_chart(bags_by_weekday ,use_container_width=True)

		#chart = (alt.Chart(sorted_week_bags).mark_bar().encode(alt.X("date"),alt.Y("Average_no_bags"),alt.Color("date:O"),alt.Tooltip(["date"]),).interactive())
		#st.altair_chart(chart)


		


		st.markdown("<h3 style='text-align: center; color: red;'>Surerewards Customers</h3>", unsafe_allow_html=True)
		metric_row({ " Total No Of Surerewards Customers ": num_reg['num_of_reg'].sum(),"Customers With PPC130 Promo Code": num_promo_reg['No_Promocode'].sum(),"Total No Of Bags": round(num_bags["number_of_bags"].sum()) })

		st.markdown("<h3 style='text-align: center; color: red;'>Receipts Upload</h3>", unsafe_allow_html=True)
		metric_row( {"Total No of Receipts Upload": num_receipts['no_of_receipts_upload'].sum(),"Total No of Valid Receipts": num_valid_receipts["no_of_valid_receipts"].sum(),"Total No of Invalid Receipts":num_invalid_receipts["no_of_invalid_receipts"].sum()})

		st.markdown("<h3 style='text-align: center; color: red;'>User Engagement</h3>", unsafe_allow_html=True)
		metric_row( {"Total No of Users With Recipets Upload": num_user_r_upload['no_of_receipts_upload'].sum(),"Total No of Users With Valid Receipts": num_user_valid_receipts["no_of_users_valid_receipts"].sum(),"Total No of Users With Invalid Receipts":num_user_invalid_receipts["no_of_users_invalid_receipts"].sum()})


		col1, col2, col3 = st.columns(3)
		col1.metric("Temperature", "70 °F", "1.2 °F")
		col2.metric("Wind", "9 mph", "-8%")
		col3.metric("Humidity", "86%", "4%")

		
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")




		if st.button("Model Information"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("The prediction model is trained on live data using pipelines for prediction accuracy")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
