from flask import Flask, render_template, request, redirect
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_distances

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':

        with open('rfr_best_jabar', 'rb') as r:
            model = pickle.load(r)

        city = request.form.get("city")
        bed = request.form.get("bed")
        bath = request.form.get("bath")
        park = request.form.get("park")
        surface = request.form.get("surface")
        building = request.form.get("building")

        datas = np.array((city, bed, bath, park, surface, building))
        datas = np.reshape(datas, (1, -1))

        pricepred = model.predict(datas)

        def getcity(x):
            if (x == str(0)):
                return "Bandung"
            elif (x == str(1)):
                return "Banjar"
            elif (x == str(2)):
                return "Bekasi"
            elif (x == str(3)):
                return "Bogor"
            elif (x == str(4)):
                return "Cianjur"
            elif (x == str(5)):
                return "Cikarang"
            elif (x == str(6)):
                return "Cimahi"
            elif (x == str(7)):
                return "Cirebon"
            elif (x == str(8)):
                return "Depok"
            elif (x == str(9)):
                return "Indramayu"
            elif (x == str(10)):
                return "Karawang"
            elif (x == str(11)):
                return "Purwakarta"
            elif (x == str(12)):
                return "Subang"
            elif (x == str(13)):
                return "Sumedang"
            elif (x == str(14)):
                return "Bandung Barat"
            elif (x == str(15)):
                return "Ciamis"
            elif (x == str(16)):
                return "Garut"
            elif (x == str(17)):
                return "Kuningan"
            elif (x == str(18)):
                return "Majalengka"
            elif (x == str(19)):
                return "Pangandaran"
            elif (x == str(20)):
                return "Sukabumi"
            elif (x == str(21)):
                return "Tasikmalaya"

        def recomendation(data):
            #convert price column
            data = pd.read_csv(data)
            data['price'] = data['price'].replace({
                ',' : '.',
                ' Juta' : '*1E6',
                ' Miliar' : '*1E9'
            }, regex = True).map(pd.eval).astype(int)
            data = data.rename(columns={'province': 'city'})
            #change city who not uncompatible
            def get_city(item):
                if ' ' in item:
                    return item[:item.find(' ')]
                else:
                    return item
            #labeling city
            data['city'] = data['city'].apply(get_city)

            #transform data
            le = LabelEncoder()
            data['city'] = le.fit_transform(data['city'])

            #recomendation data
            datarec = data[['city', 'bedroom',	'bathroom',	'parkingarea',	'surfacearea',	'buildingarea', 'price']]

            #input data
            code = [[int(city), int(bed), int(bath), int(park), int(surface), int(building), int(pricepred)]]
            #recomendation
            rec = cosine_distances(code, datarec.values)
            recidx = rec.argsort()[0, 0:100]

            recommend = data.loc[recidx]
            
            dtrecomendation = recommend[['name','price','city', 'bedroom',	'bathroom',	'parkingarea',	'surfacearea',	'buildingarea','link']].reset_index().sort_values(by='price', ascending=False)
                      
            return dtrecomendation

        def recomendation2(data):
            minprice = int(pricepred) - 100000000
            maxprice = int(pricepred) + 100000000
            minsur = int(surface) - 10
            maxsur = int(surface) + 10
            data = data[(data['price'] >= minprice) & (data['price'] <= maxprice) & 
                        (data['surfacearea'] >= minsur) & (data['surfacearea'] <= maxsur)]
            data['price'] = data['price'].map('{:,.0f}'.format)
            data = data.drop(columns = ['index'])

            return data

        dtrec = recomendation('data_clean.csv')
        dtrec2 = recomendation2(dtrec)
        dtrec2 = dtrec2.loc[dtrec2['city'] == int(city)].head()   
        strcity = {0:'Bandung', 1:'Banjar', 2:'Bekasi', 3:'Bogor', 4:'Cianjur', 5:'Cikarang', 6:'Cimahi',
                    7:'Cirebon', 8:'Depok', 9:'Indramayu', 10:'Karawang', 11:'Kuningan', 12:'Purwakarta', 13:'Subang',
                    14:'Sumedang', 15:"Bandung Barat", 16:"Ciamis", 16:"Garut", 17:"Kuningan", 18:"Majalengka", 19:"Pangandaran",
                    20: "Sukabumi", 21:"Tasikmalaya"}
        dtrec2['city'] = dtrec2['city'].map(strcity)

        name = dtrec2['name'].values
        cityrec = dtrec2['city'].values
        price = dtrec2['price'].values
        bedroom = dtrec2['bedroom'].values
        bathroom = dtrec2['bathroom'].values
        surfacearea = dtrec2['surfacearea'].values
        buildingarea = dtrec2['buildingarea'].values
        link = dtrec2['link'].values
        parkingarea = dtrec2['parkingarea'].values

        heading=('Nama Rumah', 'Harga','Daerah','Kamar Tidur', 'Kamar Mandi', 'Muatan Garasi', 'Luas Tanah', 'Luas Bangunan', 'Informasi Selengkapnya')
        

        return render_template('result.html', finalData = 'Rp. {:,.0f}'.format(int(pricepred)), city=getcity(city), bath=bath, bed=bed, surface=surface, park=park, building=building, 
        col=heading, column_names=dtrec2.columns.values, row_data=list(dtrec2.values.tolist()), zip=zip,
        name=name, cityrec=cityrec, parkingarea=parkingarea, price=price, dtrec2=[dtrec2.to_html()], bedroom=bedroom, bathroom=bathroom, surfacearea=surfacearea, buildingarea=buildingarea, link=link)

    else:
        return render_template('predict.html')

# @app.route('/detail')
# def detail():
#     return render_template('detail.html')

# @app.route('/detail2')
# def detail2():
#     return render_template('detail2.html')

# @app.route('/detail3')
# def detail3():
#     return render_template('detail3.html')

# @app.route('/detail4')
# def detail4():
#     return render_template('detail4.html')

# @app.route('/detail5')
# def detail5():
#     return render_template('detail5.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)