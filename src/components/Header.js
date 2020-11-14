import React from 'react';
import githubWhiteIcon from '../images/github_white.png';
import linkedinWhiteIcon from '../images/linkedin_white.png';
import twitterWhiteIcon from '../images/twitter_white.png';

const Header = () => {
  return (
    <header>
      <div className='header-main-wrapper'>
        <h1>Shreyas Hosamane</h1>
        <div className='social-links-wrapper'>
          <a href='https://github.com/hmshreyas7' target='_blank'>
            <img src={githubWhiteIcon} title='GitHub' alt='GitHub' />
          </a>
          <a
            href='https://www.linkedin.com/in/shreyas-hosamane/'
            target='_blank'
          >
            <img src={linkedinWhiteIcon} title='LinkedIn' alt='LinkedIn' />
          </a>
          <a href='https://twitter.com/ShreyasHosamane' target='_blank'>
            <img src={twitterWhiteIcon} title='Twitter' alt='Twitter' />
          </a>
        </div>
      </div>
      <ul className='nav-list'>
        <li>Home</li>
        <li>Projects</li>
        <li>Coursework</li>
      </ul>
    </header>
  );
};

export default Header;
