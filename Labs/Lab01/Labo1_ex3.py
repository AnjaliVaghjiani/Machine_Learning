from collections import defaultdict

def parse_birth_info(filename):
    births_by_city = defaultdict(int)
    births_by_month = defaultdict(int)
    cities = set()
    
    with open(filename, 'r') as file:
        for line in file:
            name, surname, birthplace, birthdate = line.strip().split()
            city = birthplace
            month = birthdate.split('/')[1]

            births_by_city[city] += 1
            births_by_month[month] += 1
            cities.add(city)

    return births_by_city, births_by_month, cities

months_name  = {
    1: 'January',
    2:'February',
    3:'March',
    4:'April',
    5:'May',
    6:'June',
    7:'July',
    8:'August',
    9:'September',
    10:'October',
    11:'November',
    12:'December'}

def main():
    filename = 'ex3_data.txt'
    f =open(filename, 'r')
    cities = {}
    months ={}
    for lines in f:
        name, surname, city, birthdata = lines.split()
        month = birthdata.split("/")[1]
        month = int(month) #converting string to integer
        if city not in cities:
            cities[city] =0 
        cities[city]+=1    
        
        if month not in months:
            months[month] = 0
        months[month]+=1
        
   # f.close()

    print("Births per city:")
    for city in cities:
        print('%s: %d' % (city, cities[city]))

    print("\nBirths per month:")
    
    for month_num in sorted(months):
       
        print('%s: %d' % (months_name[month_num], months[month_num]))
        

    average_births = sum(cities.values()) / len(cities.keys())
    print(f"\nAverage number of births: {average_births:.2f}")

if __name__ == "__main__":
    main()
