"""
config file to initialize the event class
"""
import obspy



"""
START AND END TIMES
"""

swarm_starttime = {
     '141':obspy.UTCDateTime('2019-05-21T07:30:00')
    ,'188':obspy.UTCDateTime('2019-07-07T08:40:00.0Z')
    ,'197':obspy.UTCDateTime('2019-07-16T21:45:00.0Z')
    ,'211':obspy.UTCDateTime('2019-07-30T22:30:05.142999Z')
    }

swarm_endtime = {
    '141':obspy.UTCDateTime('2019-05-21T09:00:00.0Z')
    ,'188':obspy.UTCDateTime('2019-07-07T10:15:00.0Z')
    ,'197':obspy.UTCDateTime('2019-07-17T00:00:00.337999Z')
    ,'211':obspy.UTCDateTime('2019-07-30T23:07:04.430999Z')
    }

# swarm_starttime = obspy.UTCDateTime('2019-05-21T07:30:00')
# swarm_endtime = obspy.UTCDateTime('2019-05-21T08:38:30')

"""
DAY 141
"""

hydrophone_metadata_141 = {
    'h1':{
        # start and end identifies the start time of the swarm where the amplitude magnitude is the highest
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        # depth of the hydrophone
        ,'depth':30
        ,'velocity_model':1750
    }
    ,    'h2':{
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        ,'depth':100        
        ,'velocity_model':1750

    }
    ,    'h3':{
        'start':obspy.UTCDateTime('2019-05-21T07:35:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T07:48:00Z')
        ,'depth':170        
        ,'velocity_model':1750

    }
    ,'h4':{
        'start':obspy.UTCDateTime('2019-05-21T07:48:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:07:00Z')
        ,'depth':240
        ,'velocity_model':1750
    }
    ,'h5':{
        'start':obspy.UTCDateTime('2019-05-21T08:07:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:34:00Z')
        # ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'depth':310
        ,'velocity_model':1750
    }
    ,'h6':{
        'start':obspy.UTCDateTime('2019-05-21T08:34:00Z')
        ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'depth':380
        ,'velocity_model':1750
    }
}


"""
DAY 188
"""
hydrophone_metadata_188 = {
    'h1':{
        # start and end identifies the start time of the swarm where the amplitude magnitude is the highest
        'start':swarm_starttime['188']
        ,'end':swarm_endtime['188']
        # depth of the hydrophone
        ,'depth':30
        ,'velocity_model':1750
    }
    ,    'h2':{
        'start':swarm_starttime['188']
        ,'end':swarm_endtime['188']
        ,'depth':100        
        ,'velocity_model':1750

    }
    ,    'h3':{
        'start':swarm_starttime['188']
        ,'end':obspy.UTCDateTime('2019-07-07T09:10:00Z')
        ,'depth':170        
        ,'velocity_model':1750

    }
    ,'h4':{
        'start':obspy.UTCDateTime('2019-07-07T09:10:00Z')
        ,'end':obspy.UTCDateTime('2019-07-07T09:30:00Z')
        ,'depth':240
        ,'velocity_model':1750
    }
    ,'h5':{
        'start':obspy.UTCDateTime('2019-07-07T09:30:00Z')
        ,'end':obspy.UTCDateTime('2019-07-07T09:45:00Z')
        # ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'depth':310
        ,'velocity_model':1750
    }
    ,'h6':{
        'start':obspy.UTCDateTime('2019-07-07T09:30:00Z')
        ,'end':swarm_endtime['188']
        ,'depth':380
        ,'velocity_model':1750
    }
}

"""
Day 197
"""
hydrophone_metadata_197 = {
    'h1':{
        # start and end identifies the start time of the swarm where the amplitude magnitude is the highest
        'start':swarm_starttime['197']
        ,'end':swarm_endtime['197']
        # depth of the hydrophone
        ,'depth':30
        ,'velocity_model':1750
    }
    ,    'h2':{
        'start':swarm_starttime['197']
        ,'end':swarm_endtime['197']
        ,'depth':100        
        ,'velocity_model':1750

    }
    ,    'h3':{
        'start':swarm_starttime['197']
        ,'end':swarm_endtime['197']
        ,'depth':170        
        ,'velocity_model':1750

    }
    ,'h4':{
        'start':swarm_starttime['197']
        ,'end':swarm_endtime['197']
        ,'depth':240
        ,'velocity_model':1750
    }
    ,'h5':{
        'start':swarm_starttime['197']
        ,'end':swarm_endtime['197']
        # ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'depth':310
        ,'velocity_model':1750
    }
    ,'h6':{
        'start':swarm_starttime['197']
        ,'end':swarm_endtime['197']
        ,'depth':380
        ,'velocity_model':1750
    }
}

"""
Day 211
"""
hydrophone_metadata_211 = {
    'h1':{
        # start and end identifies the start time of the swarm where the amplitude magnitude is the highest
        'start':swarm_starttime['211']
        ,'end':swarm_endtime['211']
        # depth of the hydrophone
        ,'depth':30
        ,'velocity_model':1750
    }
    ,    'h2':{
        'start':swarm_starttime['211']
        ,'end':swarm_endtime['211']
        ,'depth':100        
        ,'velocity_model':1750

    }
    ,    'h3':{
        'start':swarm_starttime['211']
        ,'end':obspy.UTCDateTime('2019-07-30T22:45:05.142999Z')
        ,'depth':170        
        ,'velocity_model':1750

    }
    ,'h4':{
        'start':obspy.UTCDateTime('2019-07-30T22:45:05.142999Z')
        ,'end':obspy.UTCDateTime('2019-07-30T22:52:05.142999Z')
        ,'depth':240
        ,'velocity_model':1750
    }
    ,'h5':{
        'start':obspy.UTCDateTime('2019-07-30T22:52:05.142999Z')
        ,'end':obspy.UTCDateTime('2019-07-30T22:58:05.142999Z')
        # ,'end':obspy.UTCDateTime('2019-5-21T08:38:00Z')
        ,'depth':310
        ,'velocity_model':1750
    }
    ,'h6':{
        'start':obspy.UTCDateTime('2019-07-30T22:58:05.142999Z')
        ,'end':swarm_endtime['211']
        ,'depth':380
        ,'velocity_model':1750
    }
}

"""
COMBINE EVERYTHING
"""
hydrophone_metadata = {
    '141':hydrophone_metadata_141
    ,'188':hydrophone_metadata_188
    ,'197':hydrophone_metadata_197
    ,'211':hydrophone_metadata_211
    }