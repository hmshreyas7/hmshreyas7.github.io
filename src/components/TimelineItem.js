import React from 'react';

const TimelineItem = (props) => {
  return (
    <div className='timeline'>
      <div className='period'>{props.period}</div>
      <div className='info'>
        {!props.isLeft && <div className='desc'>{props.desc}</div>}
        <div className='logo'>
          <img width='400px' src={props.logo} alt='Company/University Logo' />
        </div>
        {props.isLeft && <div className='desc'>{props.desc}</div>}
      </div>
    </div>
  );
};

export default TimelineItem;
