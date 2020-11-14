import React from 'react';
import TimelineItem from './TimelineItem';
import timelineData from '../data/timelineData';

const generateTimelineItems = (item, index) => {
  return (
    <TimelineItem
      key={item.period}
      period={item.period}
      desc={item.desc}
      logo={item.logo}
      isLeft={index % 2 === 0}
    />
  );
};

const Timeline = () => {
  return (
    <div className='timeline-wrapper'>
      <p className='welcome'>Welcome to my front page on the internet!</p>
      <div className='content'>{timelineData.map(generateTimelineItems)}</div>
    </div>
  );
};

export default Timeline;
