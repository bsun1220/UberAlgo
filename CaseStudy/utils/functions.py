from utils.constants import *

def match(find_closest, find_driver, find_path, lim_start = 0, 
          lim_end = 5000, init = True, prob_stay = PROB_STAY, skip_no_match = True):
    
    passengers, drivers, adj, nodes = initialize_data(True)
    edited_passengers, edited_drivers = [], []
    if not init:
        for p in passengers:
            start_coord = {'lat':p[1], 'lon':p[2]}
            start_node = find_closest(start_coord, nodes)
            end_coord = {'lat':p[3], 'lon':p[4]}
            end_node = find_closest(end_coord, nodes)

            row = (p[0], start_node, end_node, p[5], p[6], p[7])
            edited_passengers.append(row)
            
        with open("data/edited_passengers.pkl", "wb") as outfile:
            pickle.dump(edited_passengers, outfile)
        
        for d in drivers:
            start_coord = {'lat':d[1], 'lon':d[2]}
            start_node = find_closest(start_coord, nodes)
            row = (d[0], start_node, d[3], d[4], d[5])
            edited_drivers.append(row)
        
        with open("data/drivers.pkl", "wb") as outfile:
             pickle.dump(edited_passengers, outfile)
        
    else:
        edited_passengers = pd.read_pickle('data/edited_passengers.pkl')
        edited_drivers = pd.read_pickle('data/edited_drivers.pkl')

    res_customer, res_driver, res_time = [], [], [0]
    
    for index in range(lim_start, lim_end):
        
        passenger = edited_passengers[index]
        
        if edited_drivers[0][0] > passenger[0] and skip_no_match:
            continue
        
        time_start = time.time()
        
        driver, num = find_driver(passenger, edited_drivers, nodes, adj, find_path)
        driver_node = driver[1]
        source_node = passenger[1]
        dest_node = passenger[2]

        weekday, arrival_hour = driver[2], driver[3]
        if passenger[0] > driver[0]:
            weekday, arrival_hour = passenger[3], passenger[4]
        
        col = get_edge_num(weekday, arrival_hour)

        time_to_passenger = find_path(driver_node, source_node, adj, nodes, col)
        
        wait_time = max(driver[0] - passenger[0], 0)
        
        arrival_hour = int(arrival_hour + time_to_passenger/60) % 24
        col = get_edge_num(weekday, arrival_hour)
        
        time_to_dest = find_path(source_node, dest_node, adj, nodes, col)
        
        if time_to_dest == -1 or time_to_passenger == -1:
            continue
        
        res_customer.append(time_to_passenger + wait_time)
        res_driver.append(time_to_dest - time_to_passenger)
        edited_drivers.pop(num)
        
        if np.random.random() < prob_stay:
            new_time = int(time_to_passenger + time_to_dest + wait_time + passenger[0])
            new_hour = int(passenger[4] + time_to_passenger/60 + time_to_dest/60 + wait_time/60) % 24
            new_driver = (new_time, passenger[2], driver[2], new_hour, driver[4])
            bisect.insort(edited_drivers, new_driver, key=lambda x: x[0])
        
        time_end = time.time()
        res_time.append(res_time[-1] + time_end - time_start)
        
        if len(edited_drivers) == 0:
            break
    return np.array(res_customer), np.array(res_driver), np.array(res_time)


def process_results(res_cus, res_driv, res_time):
    plt.rcParams['figure.figsize'] = [12, 4]
    fig, ax = plt.subplots(1, 3)
    fig.tight_layout(pad = 1.5)
    ax[0].hist(res_cus)
    
    ax[0].set_xlabel(f'''Mean : {np.round(res_cus.mean(), 2)}
    Median : {np.round(np.quantile(res_cus, 0.5), 2)}
    Std : {np.round(res_cus.std(), 2)}
    ''')
    
    ax[0].set_ylabel('Count')
    ax[0].set_title(f'Customer Waiting Time (Min)')
    ax[1].hist(res_driv)
    ax[1].set_xlabel(f'''Mean : {np.round(res_driv.mean(), 2)}
    Median : {np.round(np.quantile(res_driv, 0.5), 2)}
    Std : {np.round(res_driv.std(), 2)}
    ''')
    ax[1].set_ylabel('Count')
    ax[1].set_title(f'Driver Profit (Min)')
    
    
    ax[2].plot(np.arange(len(res_time)), res_time)
    ax[2].set_title("Run Time")
    ax[2].set_xlabel('Passengers')
    ax[2].set_ylabel("Cumulative Time")
    
    plt.show()