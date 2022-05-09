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
import datetime



## Writing the latest time update
from datetime import datetime
import pytz

# Vectorizer

#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file




# The main function where we will build the actual app

# Connecting To The Database Using MySQL
conn=pymysql.connect(host='us-mm-dca-d04d6c3c8e49.g5.cleardb.net',port=int(3306),user='b04c9045b8a037',passwd='f9f69807ebbcc01',db='heroku_c6143c8aee66786')


## Setting The Content To fill the Page
st.set_page_config(layout="wide")


def main():

	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages


	# Loading Heading Images 


	## Function that automatically load images
	from PIL import Image



	
		
	st.sidebar.image("resources/imgs/surerewards.png", use_column_width=True)
	
	def load_image(image_file):
		img = Image.open(image_file)
		return img
	
	col1, col2, col3 = st.columns([1,20,1])

	with col1:
		st.write("")

	with col2:
		st.image("resources/imgs/Heading_Logo.PNG", use_column_width=True)
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
	
	
	







	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	options = ["Surerewards Insights","Prediction Page"]
	selection = st.sidebar.selectbox("Select Page", options)





	# Building out the "Information" page
	if selection == "Prediction Page":

		
		
		st.markdown("<h3 style='text-align: center; color: black;'>This Page focuses on predicting number of bag and setting dynamic daily targets</h3>", unsafe_allow_html=True)




		num_bags =pd.read_sql_query("select cast(rdata.updatedAt as date) as date, sum(ppc_surebuild + ppc_surecast + ppc_surecem + ppc_suretech + ppc_surewall + ppc_plaster + ppc_motor + ppc_sureroad) as number_of_bags from receipts as r inner join receiptdata as rdata on r.id =rdata.receipt_id where r.status in ('approved','Limit reached.') and cast(rdata.updatedAt as date)  >= '2022-02-15' and cast(rdata.updatedAt as date) < current_date() group by date" ,conn)
		
		content = {'date': list(pd.to_datetime(num_bags['date'], errors='coerce')),'number_of_bags': list(num_bags['number_of_bags'])}
		df2 = pd.DataFrame(content).set_index('date')
		df2.sort_index(inplace=True)

		from statsmodels.tsa.seasonal import seasonal_decompose

		st.info("The number of bags by date is found to consist of two hidden properties, after decomposition the first property is a trend that shows that the number of bags bought by surerewards customers is increasing with time. The second property is that the number of bags  bought by surerewards customers has a weekly seasonality, that is it varies depending on the day of the week (with the weekend having the lowest sales ")


		if st.checkbox('Show trend'): # data is hidden if box is unchecked
			ax = seasonal_decompose(df2['number_of_bags'],period =7)

			fig =ax.plot()


			fig.set_size_inches((12, 9))
			# Tight layout to realign things
			fig.tight_layout()
			#plt.show()

			#results.plot();
			st.pyplot(fig)
		st.markdown("<h5 style='text-align: center; color: black;'>How the model works</h5>", unsafe_allow_html=True)
		if st.checkbox('Show How the works'):
		
			st.info("The model trained is LSTM ( Long Short Term Memory) model,which falls under Recurrent Neural Network models. This model is a Time Series Model that uses historical data for forecasting/ predicting the number of bags in the upcoming 7 days")			
		
		if st.button("Train Model"):

			train = df2.iloc[:len(df2)]
			test = df2.iloc[len(df2)-7:]


			## Scaling the Data
			from sklearn.preprocessing import MinMaxScaler
			scaler = MinMaxScaler()

			scaler.fit(train)
			scaled_train = scaler.transform(train)
			scaled_test = scaler.transform(test)

			#Pre processing
			from keras.preprocessing.sequence import TimeseriesGenerator

			# spliting into 7 days splits
			n_input = 7
			n_features = 1
			generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


			from keras.models import Sequential
			from keras.layers import Dense
			from keras.layers import LSTM

			# define model
			model = Sequential()
			model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
			model.add(Dense(1))
			model.compile(optimizer='adam', loss='mse')


			# fit model
			model.fit(generator,epochs=50)

			st.markdown("<h5 style='text-align: center; color: black;'>Model Convergence</h5>", unsafe_allow_html=True)
			fig, ax = plt.subplots(figsize=(10,5))
			plt.tight_layout()
			loss_per_epoch = model.history.history['loss']
			ax=plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
			plt.xlabel('Number of iterations')
			plt.ylabel('means squeared error')
			st.pyplot(fig)


			#making predictions
			test_predictions = []

			first_eval_batch = scaled_train[-n_input:]
			current_batch = first_eval_batch.reshape((1, n_input, n_features))


			test_predictions = []

			first_eval_batch = scaled_train[-n_input:]
			current_batch = first_eval_batch.reshape((1, n_input, n_features))

			for i in range(len(test)):
    
				# get the prediction value for the first batch
				current_pred = model.predict(current_batch)[0]
    
				# append the prediction into the array
				test_predictions.append(current_pred) 
    
				# use the prediction to update the batch and remove the first value
				current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)



			
			## Making Predictions
			true_predictions = scaler.inverse_transform(test_predictions)
				
			## Last day of the 
			last_day = train.index[len(list(train['number_of_bags']))-1]
			last_day=last_day.date()

			import datetime
			forward_days=[last_day+datetime.timedelta(days=1)]
			for i in range(6):    
				forward_days.append(forward_days[len(forward_days)-1]+datetime.timedelta(days=1))

			## Prediction Values
			pred_values=[]
			for i in range(len(true_predictions)):
				pred_values.append(round(true_predictions.tolist()[i][0]))

				
			## Prediction Dataframe
			content = {'date': list(pd.to_datetime(forward_days, errors='coerce')),'number_of_bags': pred_values}
			df_pred = pd.DataFrame(content).set_index('date')
			df_pred.sort_index(inplace=True)
				
			## Show dataframe prediction page
			st.markdown("<h5 style='text-align: center; color: black;'>Predicted Values </h5>", unsafe_allow_html=True)
			#st.dataframe(df_pred)

			## Insert join prediction and train dataframe
			df_pred.loc[pd.to_datetime(train.index[len(train)-1], errors='coerce')] = round(train['number_of_bags'][len(train)-1])
				
			fig, ax = plt.subplots(figsize=(10,5))
			plt.tight_layout()
			labels=['historical data','prediction']
			for i,df in enumerate([train,df_pred],1):
				df =df.sort_index()
				ax = plt.plot(df.index,df['number_of_bags'],label=labels[i-1])

			plt.legend()
			plt.xlabel('Date')
			plt.ylabel('Number Of Bags')
			plt.show()
			st.pyplot(fig)

			st.success("Success") 
			








			






		

		
	
	
	# Building out the predication page
	if selection == "Surerewards Insights":
		st.markdown("<h1 style='text-align: center; color: red;'>PPC130 Surerewards Insights</h1>", unsafe_allow_html=True)

		st.info("The following visuals are based on lived data from the surerewards platform (They change with time)")

		## Show the latest Update Time
		from datetime import datetime

		SA_time = pytz.timezone('Africa/Johannesburg') 
		datetime_SA = datetime.now(SA_time)
		metric("Latest Time Update", datetime_SA.strftime('%Y-%m-%d %H:%M %p'))
		
		
	

		# Loaading Datasets
		num_bags =pd.read_sql_query("select cast(rdata.updatedAt as date) as date,sum(ppc_surebuild + ppc_surecast + ppc_surecem + ppc_suretech + ppc_surewall + ppc_plaster + ppc_motor + ppc_sureroad) as number_of_bags from receipts as r inner join receiptdata as rdata on r.id =rdata.receipt_id where r.status in ('approved','Limit reached.') and cast(rdata.updatedAt as date)  >= '2022-02-15' and cast(rdata.updateddAt as date) < current_date() group by date" ,conn)
		
		num_reg =pd.read_sql_query(" Select count(*) as num_of_reg,cast(createdAt as date) as date  from users where cast(createdAt as date) >= '2022-02-15'  group by date order by date",conn)
		num_promo_reg = pd.read_sql_query("Select cast(createdAt as date) as date,count(*) as No_Promocode from users where code = 'PPC130' and cast(createdAt as date)>='2022-02-15'  group by date order by Date",conn)
		num_receipts=pd.read_sql_query("SELECT count(*) as no_of_receipts_upload ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Approved','Limit reached.') and cast(receipts.updatedAt as date) >= '2022-02-15' and cast(receipts.updatedAt as date) < current_date()  group by date",conn)
		num_valid_receipts=pd.read_sql_query("SELECT count(*) as no_of_valid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  Not in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Unprocessed') and cast(receipts.updatedAt as date) >= '2022-02-15' and cast(receipts.updatedAt as date) < current_date()  group by date",conn)
		num_invalid_receipts=pd.read_sql_query("SELECT count(*) as no_of_invalid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible') and cast(receipts.updatedAt as date) >= '2022-02-15' and cast(receipts.updatedAt as date) < current_date()  group by date",conn)

		num_user_r_upload=pd.read_sql_query("SELECT count(distinct(users.id)) as no_of_receipts_upload ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Approved','Limit reached.') and cast(receipts.updatedAt as date) >= '2022-02-15' and cast(receipts.updatedAt as date) < current_date()  group by date",conn)
		
		num_user_valid_receipts=pd.read_sql_query("SELECT count(distinct(users.id)) as no_of_users_valid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  Not in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible','Unprocessed') and cast(receipts.updatedAt as date) >= '2022-02-15' and cast(receipts.updatedAt as date) < current_date()  group by date",conn)
		num_user_invalid_receipts=pd.read_sql_query("SELECT count(distinct(users.id))as no_of_users_invalid_receipts ,cast(receipts.updatedAt as date) as date from users inner join receipts on users.id=receipts.user_id where status  in ('Duplicate receipts','outdated Receipt','Receipt cut off.','Receipt not relevant', 'Receipt not visible') and cast(receipts.updatedAt as date) >= '2022-02-15' and cast(receipts.updatedAt as date) < current_date()  group by date",conn)
		
		num_bags["delta"] = (num_bags["number_of_bags"].pct_change()).fillna(0)

		## Number Of Bags Visual
		st.markdown("<h4 style='text-align: center; color: black;'>The visual below shows the number of bags bought by surerewards customers.</h4>", unsafe_allow_html=True)
		st.altair_chart(line_graph(num_bags,"date","number_of_bags"), use_container_width=True)
		
		## Grouping By Weekday
		num_bags['date'] = pd.to_datetime(num_bags['date'], errors='coerce')
		num_bags['weekday'] = num_bags["date"].dt.day_name()


			
        ## Number of registration Visual
		st.markdown("<h4 style='text-align: center; color: black;'>The visual below shows the number of customer registration on the surerewards platform.</h4>", unsafe_allow_html=True)
		
		num_reg["delta"] = (num_reg["num_of_reg"].pct_change()).fillna(0)
		st.altair_chart(line_graph(num_reg ,"date","num_of_reg").interactive(), use_container_width=True)

		## Number of registration receipts upload
		st.markdown("<h4 style='text-align: center; color: black;'>The visual below shows the number of receipts upload on the surerewards platform.</h4>", unsafe_allow_html=True)
		
		num_receipts["delta"] = (num_receipts["no_of_receipts_upload"].pct_change()).fillna(0)
		st.altair_chart(line_graph(num_receipts ,"date","no_of_receipts_upload").interactive(), use_container_width=True)
       
	   
	   
		## Numbers of bags by weekday


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




 		## Number of registration Visual
		st.markdown("<h5 style='text-align: center; color: black;'>The visual below shows the average and total number of bags bought buy surerewards customer by weekday.</h5>", unsafe_allow_html=True)
		

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

		## Space 

		st.text("")
		st.text("")
		st.text("")

		## Space

		import warnings
		warnings.filterwarnings("ignore")

		
		st.markdown("<h2 style='text-align: center; color: black;'>Key Performance Indicators ( KPIs ).</h2>", unsafe_allow_html=True)


		st.markdown("<h3 style='text-align: center; color: red;'>Surerewards Customers</h3>", unsafe_allow_html=True)
		metric_row({ " Total No Of Surerewards Customers ": num_reg['num_of_reg'].sum(),"Customers With PPC130 Promo Code": num_promo_reg['No_Promocode'].sum(),"Total No Of Bags": round(num_bags["number_of_bags"].sum()) })

		st.markdown("<h3 style='text-align: center; color: red;'>Customer Receipts Upload</h3>", unsafe_allow_html=True)
		metric_row( {"Total No of Receipts Upload": num_receipts['no_of_receipts_upload'].sum(),"Total No of Valid Receipts": num_valid_receipts["no_of_valid_receipts"].sum(),"Total No of Invalid Receipts":num_invalid_receipts["no_of_invalid_receipts"].sum()})

		st.markdown("<h3 style='text-align: center; color: red;'>Customer Engagement</h3>", unsafe_allow_html=True)
		metric_row( {"Total No of Users With Recipets Upload": num_user_r_upload['no_of_receipts_upload'].sum(),"Total No of Users With Valid Receipts": num_user_valid_receipts["no_of_users_valid_receipts"].sum(),"Total No of Users With Invalid Receipts":num_user_invalid_receipts["no_of_users_invalid_receipts"].sum()})
		warnings.filterwarnings("ignore")

		#col1, col2, col3 = st.columns(3)
		#col1.metric("Temperature","70 °F",  "1.2 °F")
		#col2.metric("Wind", "9 mph", "-8%")
		#col3.metric("Humidity", "86%", "4%")




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
