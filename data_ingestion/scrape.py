import fastf1
import requests
import logging
from urllib.parse import quote
import os

'''
Code designed to intake telemetry data from the Tracing Insights GitHub repository. 
- The program prompts the user to select a year and a race from that year, then iterates through all drivers and laps for that race, downloading the telemetry data for each lap and saving it.
- The program uses the FastF1 API to obtain the schedule and session information, and the requests library to fetch the telemetry data from the GitHub repository. 
- The telemetry data is saved in a directory structure organized by event name and driver abbreviation.
'''

# Define base components used later in the program to build url and file names
base_url = 'https://raw.githubusercontent.com/TracingInsights-Archive/2025/main/'
end_url = '/Race'
file_extension = '_tel.json'
allowed_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Disables logging messages from FastF1 API
logging.disable(logging.INFO)

def main():
    # Variable declaration
    event_dict = {}
    driver_key_dict = {}
    laps_dict = {}
    year = 0
    index = 1

    # Obtaining user input for year
    print("\nTracing Inisghts Telemetry Data Fetcher\n")
    print(f'Current allowed years: {allowed_years}')
    year = int(input("Please select a year from the above range to fetch data from\n\n"))
    while (year not in allowed_years):
        year = int(input("\nError. Year not in range of valid data. Please try again.\n\n"))
    
    # Obtain the schedule from the specified year
    print()
    schedule = fastf1.get_event_schedule(year)['EventName']
    print()

    # Print all events from schedule
    for event in schedule:
        if('Grand Prix' in event):
            print(f'{index}) {event}')
            event_dict[index] = event
            index += 1
    
    # Obtaining user input for race
    race = int(input("\nPlease select a race from the above range to fetch data from\n\n"))
    while (race not in event_dict.keys()):
        race = int(input("\nError. Race not in range of valid data. Please try again.\n\n"))

    # Get specified session and load it
    session = fastf1.get_session(year, event_dict[race], 'R')
    session.load()
    event_name = session.event['EventName']

    # Iterate through all drivers and store their abbreviations
    for driver in session.drivers:
        abbr = session.get_driver(driver)['Abbreviation']
        driver_key_dict[abbr] = driver

    # Craft the URL for fetching the files from Git
    url_event_name = quote(event_name, safe='')
    url = base_url + url_event_name + end_url

    # Obtain the lap count from each driver and store it in a dict
    for abbr in driver_key_dict.keys():
        driver_key = driver_key_dict[abbr]
        driver_laps = int(session.results['Laps'][driver_key] + 0.9999)
        
        laps_dict[abbr] = driver_laps

    print(laps_dict)
    # for driver_num, driver in enumerate(driver_key_dict.keys()):

    #     print(f'\nProcessing data for driver {driver_num + 1} / {len(driver_key_dict.keys())} ({driver}):\n')

    #     # For each driver, obtain every lap of telemetry data, download, and save it
    #     for lap in range(1, laps_dict[driver] + 1):
    #         addition_url = f'/{driver}/{lap}{file_extension}'
    #         download_url = url + addition_url

    #         print(f'Processing - Driver {driver}, Lap {lap}')
    #         r = requests.get(download_url)

    #         if r.status_code == 200:
    #             path = f'telemetry/{event_name.replace(" ", "_")}/{driver}'
    #             os.makedirs(path, exist_ok=True)

    #             with open(f'{path}/{lap}{file_extension}', 'wb') as f:
    #                 f.write(r.content)

if __name__ == '__main__':
    main()
