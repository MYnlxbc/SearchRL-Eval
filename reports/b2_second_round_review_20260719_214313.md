# B2 第二轮检索人工审查

- 样本数：31
- 建议标签：有效检索 / 无效检索 / 重复检索 / 错误查询 / 答案改善 / 答案退化

## 1. HotpotQA / dev_11

- 问题：What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
- 标准答案：David Weissman
- B0：James L. Brooks（EM=0，F1=0.0）
- B1：David Weissman（EM=1，F1=1.0）
- B2：David Diamond（EM=0，F1=0.5）
- 第一轮标题：Nicolas Cage | Téa Leoni | Evolution (2001 film)
- 第二轮查询：screenwriter Evolution Nicolas Cage Téa Leoni
- 第二轮标题：Nicolas Cage | Nicolas Cage | David Diamond (screenwriter)
- 人工标签：无效检索
- 备注：

## 2. HotpotQA / dev_14

- 问题：The football manager who recruited David Beckham managed Manchester United during what timeframe?
- 标准答案：from 1986 to 2013
- B0：1999-2008（EM=0，F1=0.0）
- B1：Eric Harrison managed Manchester United during June 1981 to the 1993-94 season.（EM=0，F1=0.1333）
- B2：Eric Harrison managed Manchester United from June 1981 to the end of the 1993–94 season.（EM=0，F1=0.2353）
- 第一轮标题：Eric Harrison (footballer) | Eric Harrison (footballer) | 1993–94 Manchester United F.C. season
- 第二轮查询：Eric Harrison Manchester United recruitment timeframe David Beckham
- 第二轮标题：Eric Harrison (footballer) | Eric Harrison (footballer) | David Beckham
- 人工标签：无效检索
- 备注：

## 3. HotpotQA / dev_16

- 问题：The Vermont Catamounts men's soccer team currently competes in a conference that was formerly known as what from 1988 to 1996?
- 标准答案：the North Atlantic Conference
- B0：The Green Mountain Conference（EM=0，F1=0.3333）
- B1：Central Atlantic Collegiate Conference（EM=0，F1=0.5714）
- B2：Central Atlantic Collegiate Conference（EM=0，F1=0.5714）
- 第一轮标题：Vermont Catamounts | Vermont Catamounts | Central Atlantic Collegiate Conference
- 第二轮查询：Central Atlantic Collegiate Conference was formerly known as from 1988 to 1996?
- 第二轮标题：Central Atlantic Collegiate Conference | Central Atlantic Collegiate Conference | South Atlantic Conference
- 人工标签：
- 备注：

## 4. HotpotQA / dev_18

- 问题：Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?
- 标准答案：1969 until 1974
- B0：Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the presidency of Jimmy Carter, which served from 1977 to 1981.（EM=0，F1=0.0）
- B1：Roger O. Egeberg served as Assistant Secretary for Health and Scientific Affairs during the Nixon administration, which served from 1969 to 1974.（EM=0，F1=0.1667）
- B2：Roger O. Egeberg served as Assistant Secretary for Health and Scientific Affairs during the Nixon administration, which was from 1969 to 1971.（EM=0，F1=0.0833）
- 第一轮标题：Roger O. Egeberg | Roger O. Egeberg | Roger O. Egeberg
- 第二轮查询：What were the years of the Nixon administration when Roger O. Egeberg served as Assistant Secretary for Health and Scientific Affairs?
- 第二轮标题：Roger O. Egeberg | Roger O. Egeberg | Roger O. Egeberg
- 人工标签：
- 备注：

## 5. HotpotQA / dev_2

- 问题：What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
- 标准答案：Animorphs
- B0：The Science Fantasy Young Adult series you are referring to is likely "The Lost Fleet" by John Ringo, though it's worth noting that the concept of enslaved worlds and alien species being narrated by companion books isn't explicitly stated for this series. A more fitting match would be "The Rangers Apprentice" series by John（EM=0，F1=0.0）
- B1：Animorphs（EM=1，F1=1.0）
- B2：Animorphs（EM=1，F1=1.0）
- 第一轮标题：Animorphs | Foreigner universe | Young Wizards
- 第二轮查询：Animorphs Companion Books Alien Species
- 第二轮标题：Animorphs | Animorphs | The Andalite Chronicles
- 人工标签：
- 备注：

## 6. HotpotQA / dev_24

- 问题：What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?
- 标准答案：World's Best Goalkeeper
- B0：The father of Kasper Schmeichel, Peter Schmeichel, was voted World's Best Goalkeeper by the IFFHS in 1992.（EM=0，F1=0.3333）
- B1：Peter Schmeichel was voted the IFFHS World's Best Goalkeeper in 1992.（EM=0，F1=0.4615）
- B2：Peter Schmeichel（EM=0，F1=0.0）
- 第一轮标题：Peter Schmeichel | Kasper Schmeichel | Peter Schmeichel
- 第二轮查询：Peter Schmeichel IFFHS vote 1992
- 第二轮标题：Peter Schmeichel | Peter Schmeichel | Peter Schmeichel
- 人工标签：
- 备注：

## 7. HotpotQA / dev_28

- 问题：Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?
- 标准答案：Henry J. Kaiser
- B0：Andrew Carnegie（EM=0，F1=0.0）
- B1：Henry J. Kaiser（EM=1，F1=1.0）
- B2：Henry J. Kaiser（EM=1，F1=1.0）
- 第一轮标题：Henry J. Kaiser | Henry J. Kaiser | Kaiser Shipyards
- 第二轮查询：Henry J. Kaiser founder of Kaiser Ventures
- 第二轮标题：Henry J. Kaiser | Henry J. Kaiser | Henry J. Kaiser
- 人工标签：
- 备注：

## 8. HotpotQA / dev_30

- 问题：When was Poison's album "Shut Up, Make Love" released?
- 标准答案：2000
- B0：Poison's album "Shut Up, Make Love" was released on November 14, 1995.（EM=0，F1=0.0）
- B1：The album "Shut Up, Make Love" was released on March 14, 2000.（EM=0，F1=0.1667）
- B2：Poison's album "Shut Up, Make Love" was released on March 14, 2000.（EM=0，F1=0.1538）
- 第一轮标题：Poison (American band) | Open Up and Say... Ahh! | Poison (American band)
- 第二轮查询：Shut Up, Make Love Poison album release date
- 第二轮标题：Poison (American band) | Dispose (album) | Honey (Scandal album)
- 人工标签：
- 备注：

## 9. HotpotQA / dev_36

- 问题：Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?
- 标准答案：2000
- B0：2005（EM=0，F1=0.0）
- B1：2000（EM=1，F1=1.0）
- B2：2000（EM=1，F1=1.0）
- 第一轮标题：Carlo Rovelli | Carlo Rovelli | Charles Fabry
- 第二轮查询：What year did Carlo Rovelli start working in France?
- 第二轮标题：Carlo Rovelli | Carlo Rovelli | Carlo Rovelli
- 人工标签：
- 备注：

## 10. HotpotQA / dev_37

- 问题：The Livesey Hal War Memorial commemorates the fallen of which war, that had over 60 million casualties?
- 标准答案：World War II
- B0：World War II（EM=1，F1=1.0）
- B1：World War I and World War II（EM=0，F1=0.6）
- B2：World War I and World War II（EM=0，F1=0.6）
- 第一轮标题：Livesey Hall War Memorial | Livesey Hall War Memorial | Livesey Hall War Memorial
- 第二轮查询：World War I casualties of Livesey Hall War Memorial memorial
- 第二轮标题：Livesey Hall War Memorial | Livesey Hall War Memorial | Livesey Hall War Memorial
- 人工标签：
- 备注：

## 11. HotpotQA / dev_39

- 问题：Ralph Hefferline was a psychology professor at a university that is located in what city?
- 标准答案：New York City
- B0：Raleigh（EM=0，F1=0.0）
- B1：Columbia University（EM=0，F1=0.0）
- B2：New York City（EM=1，F1=1.0）
- 第一轮标题：Ralph Hefferline | Ralph Hefferline | Carroll C. Pratt
- 第二轮查询：Columbia University location
- 第二轮标题：Columbia University | Columbia University | Columbia University
- 人工标签：
- 备注：

## 12. HotpotQA / dev_44

- 问题：Alfred Balk served as the secretary of the Committee on the Employment of Minority Groups in the News Media under which United States Vice President?
- 标准答案：Nelson Rockefeller
- B0：Alfred Balk served as the secretary of the Committee on the Employment of Minority Groups in the News Media under Dan Quayle, who was the Vice President of the United States from 1989 to 1993.（EM=0，F1=0.0）
- B1：Alfred Balk served as the secretary of the Committee on the Employment of Minority Groups in the News Media under the United States Vice President Nelson Rockefeller.（EM=0，F1=0.1667）
- B2：Nelson Rockefeller（EM=1，F1=1.0）
- 第一轮标题：Alfred Balk | Alfred Balk | Office of the Vice President of the United States
- 第二轮查询：Office of the Vice President of the United States: Who served as Vice President when Alfred Balk served as secretary of the Committee on the Employment of Minority Groups in the
- 第二轮标题：Office of the Vice President of the United States | Alfred Balk | Vice President of the United States
- 人工标签：
- 备注：

## 13. HotpotQA / dev_5

- 问题：2014 S/S is the debut album of a South Korean boy group that was formed by who?
- 标准答案：YG Entertainment
- B0：2014 S/S is the debut album of iKON, which was formed by JYP Entertainment.（EM=0，F1=0.1333）
- B1：MarblePop Entertainment（EM=0，F1=0.5）
- B2：YG Entertainment（EM=1，F1=1.0）
- 第一轮标题：HeartB | 2014 S/S | 2014 S/S
- 第二轮查询：Who formed the South Korean boy group that released the debut album 2014 S/S?
- 第二轮标题：HeartB | William Ellis School | 2014 S/S
- 人工标签：
- 备注：

## 14. HotpotQA / dev_9

- 问题：Are Local H and For Against both from the United States?
- 标准答案：yes
- B0：Yes, both Local H and For Against are from the United States.（EM=0，F1=0.1667）
- B1：No, Local H is from the United States, while For Against is from Nebraska, United States.（EM=0，F1=0.0）
- B2：No, Local H is from the United States, while For Against is also from the United States.（EM=0，F1=0.0）
- 第一轮标题：For Against | For Against | Local H
- 第二轮查询：Are Local H and For Against both from the United States?
- 第二轮标题：For Against | For Against | Local H
- 人工标签：
- 备注：

## 15. NQ / test_1

- 问题：when is the next deadpool movie being released?
- 标准答案：May 18, 2018
- B0：The next Deadpool movie, titled "Deadpool 3", is not scheduled for a release date as of now.（EM=0，F1=0.0）
- B1：The next Deadpool movie after "Deadpool 2" is not mentioned, and there are no upcoming Deadpool movies listed in the provided documents.（EM=0，F1=0.0）
- B2：The next Deadpool movie, Deadpool 2, was released on May 18, 2018. There is no information about a subsequent Deadpool movie in the provided documents.（EM=0，F1=0.24）
- 第一轮标题：Deadpool (film) | Deadpool 2 | Deadpool 2
- 第二轮查询：when is the next deadpool movie being released?
- 第二轮标题：Deadpool (film) | Deadpool 2 | Deadpool 2
- 人工标签：
- 备注：

## 16. NQ / test_10

- 问题：what is the most current adobe flash player version?
- 标准答案：28.0.0.137
- B0：The most current Adobe Flash Player version as of 2023 is Flash Player 29.（EM=0，F1=0.0）
- B1：The most current Adobe Flash Player version mentioned in the provided documents is 24.（EM=0，F1=0.0）
- B2：The most current Adobe Flash Player version as of the information provided is 24.（EM=0，F1=0.0）
- 第一轮标题：Adobe Flash | Adobe Flash Player | Adobe Flash Player
- 第二轮查询：what is the most recent version of adobe flash player for linux?
- 第二轮标题：Adobe Flash Player | Adobe Flash | Adobe Flash Player
- 人工标签：
- 备注：

## 17. NQ / test_11

- 问题：swan lake the sleeping beauty and the nutcracker are three famous ballets by?
- 标准答案：Pyotr Ilyich Tchaikovsky
- B0：Pyotr Ilyich Tchaikovsky（EM=1，F1=1.0）
- B1：Nevada Ballet Theatre（EM=0，F1=0.0）
- B2：George Balanchine（EM=0，F1=0.0）
- 第一轮标题：Nevada Ballet Theatre | Vienna Festival Ballet | Oakville Ballet Company
- 第二轮查询：George Balanchine
- 第二轮标题：George Balanchine | George Balanchine | George Balanchine
- 人工标签：
- 备注：

## 18. NQ / test_14

- 问题：who designed the garden city of new earswick?
- 标准答案：planner Raymond Unwin | architect Barry Parker | Raymond Unwin
- B0：The garden city of New Earswick was designed by Ebenezer Howard.（EM=0，F1=0.0）
- B1：Barry Parker（EM=0，F1=0.8）
- B2：Barry Parker（EM=0，F1=0.8）
- 第一轮标题：New Earswick | Ebenezer Howard | Garden city movement
- 第二轮查询：Barry Parker designed New Earswick garden city?
- 第二轮标题：New Earswick | Garden city movement | Richard Barry Parker
- 人工标签：
- 备注：

## 19. NQ / test_15

- 问题：what is the first step in the evolution of the eye?
- 标准答案：photoreceptor proteins that sense light | eyespots
- B0：The first step in the evolution of the eye was likely the development of light-sensitive cells (rods and cones) in the photoreceptor layer of the retina.（EM=0，F1=0.08）
- B1：The first step in the evolution of the eye, according to the provided documents, was complex, image-forming eyes appearing during the Cambrian explosion, which occurred within a few million years of the Cambrian period around 540 million years ago.（EM=0，F1=0.0）
- B2：The first step in the evolution of the eye, according to the provided documents, was the appearance of complex, image-forming eyes within the few million years of the Cambrian explosion.（EM=0，F1=0.0）
- 第一轮标题：Evolution of the eye | Evolution of the eye | Eye
- 第二轮查询：what is the first step in the evolution of the eye?
- 第二轮标题：Evolution of the eye | Evolution of the eye | Eye
- 人工标签：
- 备注：

## 20. NQ / test_23

- 问题：what part of the brain is in the middle cranial fossa?
- 标准答案：pituitary gland | the temporal lobes | the pituitary gland | temporal lobes
- B0：The middle cranial fossa contains parts of the brain including the pituitary gland and the optic nerve.（EM=0，F1=0.2667）
- B1：The temporal lobes of the brain are located in the middle cranial fossa.（EM=0，F1=0.3333）
- B2：Temporal lobes of the brain（EM=0，F1=0.6667）
- 第一轮标题：Middle cranial fossa | Middle cranial fossa | Middle cranial fossa
- 第二轮查询：what part of the brain is in the middle cranial fossa?
- 第二轮标题：Middle cranial fossa | Middle cranial fossa | Middle cranial fossa
- 人工标签：
- 备注：

## 21. NQ / test_24

- 问题：where do the greasers live in the outsiders?
- 标准答案：Tulsa, Oklahoma
- B0：In The Outsiders, the greasers primarily live in the lower socioeconomic area of Tulsa, Oklahoma, known as the East Side.（EM=0，F1=0.2222）
- B1：Tulsa, Oklahoma（EM=1，F1=1.0）
- B2：Tulsa, Oklahoma（EM=1，F1=1.0）
- 第一轮标题：The Outsiders (film) | The Outsiders (novel) | The Outsiders (novel)
- 第二轮查询：Tulsa, Oklahoma
- 第二轮标题：Tulsa, Oklahoma | Tulsa, Oklahoma | Tulsa, Oklahoma
- 人工标签：
- 备注：

## 22. NQ / test_26

- 问题：where do they grow hops in the us?
- 标准答案：Yakima (Washington) | Idaho | Washington | western Canyon County, Idaho | Willamette (Oregon) | Oregon
- B0：Maine, Idaho, Oregon, and California are major producers of hops in the US.（EM=0，F1=0.1538）
- B1：The principal regions in the US where hops are grown commercially are the Yakima Valley in Washington and the Willamette Valley in Oregon.（EM=0，F1=0.1905）
- B2：The principal production centers for hops in the US are the Yakima Valley in Washington and the Willamette Valley in Oregon.（EM=0，F1=0.2105）
- 第一轮标题：Hops | Hops | Hops
- 第二轮查询：WHERE DO THEY GROW HOPS IN THE US TODAY?
- 第二轮标题：Hops | Hops | Hops
- 人工标签：
- 备注：

## 23. NQ / test_28

- 问题：in which regions are most of africa petroleum and natural gas found?
- 标准答案：Nigeria
- B0：Most of Africa's petroleum and natural gas are found in the North African region, particularly in countries like Algeria, Libya, and Egypt. Sub-Saharan Africa also contains significant reserves, notably in Nigeria, Angola, and Equatorial Guinea.（EM=0，F1=0.0571）
- B1：Most of Africa's petroleum and natural gas are found in North Africa, specifically in countries like Libya, Algeria, and Egypt, which together account for 85% of the continent's oil production.（EM=0，F1=0.0）
- B2：Most of Africa's petroleum and natural gas are found in North Africa, with countries like Libya, Algeria, Nigeria, and Egypt dominating the oil and gas production.（EM=0，F1=0.0769）
- 第一轮标题：Mineral industry of Africa | Natural resources of Africa | Energy in Africa
- 第二轮查询：in which regions are most of africa's petroleum and natural gas found?
- 第二轮标题：Mineral industry of Africa | Natural resources of Africa | Energy in Africa
- 人工标签：
- 备注：

## 24. NQ / test_29

- 问题：in which sea pearl is found in india?
- 标准答案：the Indian Ocean
- B0：Bay of Bengal（EM=0，F1=0.0）
- B1：Gulf of Mannar（EM=0，F1=0.0）
- B2：Gulf of Mannar（EM=0，F1=0.0）
- 第一轮标题：Pearl | Industry in ancient Tamil country | Pearl Fishery Coast
- 第二轮查询：which sea is pearl found in india?
- 第二轮标题：Pearl | Pearl Fishery Coast | Industry in ancient Tamil country
- 人工标签：
- 备注：

## 25. NQ / test_35

- 问题：who played the mother in the black stallion?
- 标准答案：Teri Garr
- B0：Dorothy McGuire played the mother in The Black Stallion.（EM=0，F1=0.0）
- B1：Teri Garr（EM=1，F1=1.0）
- B2：Teri Garr played the mother in The Black Stallion.（EM=0，F1=0.4444）
- 第一轮标题：The Black Stallion Returns | The Black Stallion (film) | The Black Stallion (film)
- 第二轮查询：Teri Garr
- 第二轮标题：Teri Garr | Teri Garr | Teri Garr filmography
- 人工标签：
- 备注：

## 26. NQ / test_39

- 问题：where does the story the great gatsby take place?
- 标准答案：Long Island of 1922
- B0：The Great Gatsby takes place primarily in Long Island, New York, and West Egg, on the eastern end of the island, during the summer of 1922.（EM=0，F1=0.3077）
- B1：The story takes place on the fictional towns of West Egg and East Egg on prosperous Long Island in 1922.（EM=0，F1=0.3636）
- B2：The story takes place on the fictional towns of West Egg and East Egg on prosperous Long Island in the summer of 1922.（EM=0，F1=0.3333）
- 第一轮标题：The Great Gatsby | The Great Gatsby | The Great Gatsby (2013 film)
- 第二轮查询：The Great Gatsby setting
- 第二轮标题：The Great Gatsby | The Great Gatsby | The Great Gatsby (2013 film)
- 人工标签：
- 备注：

## 27. NQ / test_43

- 问题：what is the corporate tax rate in great britain?
- 标准答案：20%
- B0：The corporate tax rate in Great Britain is 19% for companies paying the full rate, with a reduced rate of 17% for low and medium profits.（EM=0，F1=0.0）
- B1：The corporate tax rate in Great Britain is currently 20%.（EM=0，F1=0.2）
- B2：The corporate tax rate in Great Britain is currently 20%, effective from April 2015.（EM=0，F1=0.1429）
- 第一轮标题：United Kingdom corporation tax | Taxation in the United Kingdom | United Kingdom corporation tax
- 第二轮查询：what is the current corporate tax rate in great britain?
- 第二轮标题：United Kingdom corporation tax | United Kingdom corporation tax | Business rates in England
- 人工标签：
- 备注：

## 28. NQ / test_45

- 问题：what network is showing the monday night football game?
- 标准答案：ESPN
- B0：Fox（EM=0，F1=0.0）
- B1：ABC（EM=0，F1=0.0）
- B2：ABC（EM=0，F1=0.0）
- 第一轮标题：Monday Night Football | Monday Night Football | Monday Night Football
- 第二轮查询：what network is showing the monday night football game?
- 第二轮标题：Monday Night Football | Monday Night Football | Monday Night Football
- 人工标签：
- 备注：

## 29. NQ / test_47

- 问题：what was the city of beijing previously known as?
- 标准答案：Peking
- B0：Peking（EM=1，F1=1.0）
- B1：Jingshi（EM=0，F1=0.0）
- B2：The city of Beijing was previously known as Jingshi.（EM=0，F1=0.0）
- 第一轮标题：History of Beijing | Beijing | Beijing
- 第二轮查询：Jingshi
- 第二轮标题：Jingye Temple (Shaanxi) | Jingjiang | Jingzhou
- 人工标签：
- 备注：

## 30. NQ / test_48

- 问题：when is the fourth movie of the divergent series coming out?
- 标准答案：never made
- B0：The fourth movie of the Divergent series, "Insurgent", was released on March 21, 2014.（EM=0，F1=0.0）
- B1：June 2017（EM=0，F1=0.0）
- B2：June 2017（EM=0，F1=0.0）
- 第一轮标题：The Divergent Series | The Divergent Series: Allegiant | The Divergent Series: Insurgent
- 第二轮查询：when is the fourth movie of the divergent series coming out?
- 第二轮标题：The Divergent Series | The Divergent Series: Allegiant | The Divergent Series: Insurgent
- 人工标签：
- 备注：

## 31. NQ / test_7

- 问题：when is the next scandal episode coming out?
- 标准答案：February 1, 2018
- B0：The next episode of Scandal airs on ABC on Wednesdays at 9:00 PM ET/PT.（EM=0，F1=0.0）
- B1：The next episode of Scandal has already concluded with "Over a Cliff" airing on April 19, 2018. There are no upcoming episodes scheduled as the show ended with its seventh season finale.（EM=0，F1=0.0625）
- B2：The next episode of Scandal has already aired. The series finale "Over a Cliff" concluded the seventh season on April 19, 2018.（EM=0，F1=0.0952）
- 第一轮标题：Scandal (TV series) | Over a Cliff | Scandal (season 6)
- 第二轮查询：when is the next scandal episode coming out?
- 第二轮标题：Scandal (TV series) | Over a Cliff | Scandal (season 6)
- 人工标签：
- 备注：

