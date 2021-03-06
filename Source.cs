using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;

namespace Leetcode
{
    public class Source
    {
        /// <summary>
        /// 1232. 缀点成线
        /// Easy
        /// </summary>
        /// <param name="coordinates"></param>
        /// <returns></returns>
        public bool CheckStraightLine(int[][] coordinates)
        {
            for (int i = 2; i < coordinates.Length; i++)
            {
                if ((coordinates[1][1] - coordinates[0][1]) * (coordinates[i][0] - coordinates[0][0])
                    != (coordinates[i][1] - coordinates[0][1]) * (coordinates[1][0] - coordinates[0][0]))
                {
                    return false;
                }
            }
            return true;
        }


        #region 第 224 场周赛
        public int TupleSameProduct2(int[] nums)
        {
            int len = nums.Length;
            int count = 0;
            Dictionary<int, int> dict = new Dictionary<int, int>();
            for (int i = 0; i < len; i++)
            {
                for (int j = i + 1; j < len; j++)
                {
                    var tmp = nums[i] * nums[j];
                    if (dict.ContainsKey(tmp))
                    {
                        dict[tmp]++;
                    }
                    else
                    {
                        dict.Add(tmp, 1);
                    }
                }
            }
            foreach (var item in dict)
            {
                if (item.Value > 1)
                {
                    count += item.Value * (item.Value - 1) / 2 * 8;
                }
            }
            return count;
        }


        /// <summary>
        /// 5243. 同积元组
        /// Medium
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int TupleSameProduct(int[] nums)
        {
            Array.Sort(nums);
            int n = nums.Length;
            int count = 0;
            foreach (var a in nums)
            {
                foreach (var b in nums)
                {
                    if (b == a)
                    {
                        continue;
                    }
                    foreach (var c in nums)
                    {
                        if (c == b || c == a)
                        {
                            continue;
                        }
                        foreach (var d in nums)
                        {
                            if (d == c || d == b || d == a)
                            {
                                continue;
                            }
                            if (a * b == c * d)
                            {
                                count++;
                            }
                        }
                    }
                }
            }
            return count;

            // bool Target(int[] nums)
            // {
            //     if (nums == null) return false;
            //     if (nums.Length < 4 || nums.Length > 4) return false;
            //     return nums[0] * nums[1] == nums[2] * nums[3];
            // }
        }




        /// <summary>
        /// 5653. 可以形成最大正方形的矩形数目
        /// Easy
        /// </summary>
        /// <param name="rectangles"></param>
        /// <returns></returns>
        public int CountGoodRectangles(int[][] rectangles)
        {
            if (rectangles.Length == 0) return 0;
            if (rectangles.Length == 1) return 1;
            Dictionary<int, int> dict = new Dictionary<int, int>();
            for (int i = 0; i < rectangles.Length; i++)
            {
                int tmp = Math.Min(rectangles[i][0], rectangles[i][1]);
                if (dict.ContainsKey(tmp))
                {
                    dict[tmp]++;
                }
                else
                {
                    dict.Add(tmp, 1);
                }
            }
            int res = 0;
            foreach (var item in dict)
            {
                res = Math.Max(res, item.Key);
            }
            return dict[res];
        }



        #endregion


        #region 第 223 场周赛
        /// <summary>
        /// 5639. 完成所有工作的最短时间
        /// Hard
        /// </summary>
        /// <param name="jobs"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public int MinimumTimeRequired(int[] jobs, int k)
        {
            int[,] dp = new int[13, 1 << 12];
            int[] sum = new int[1 << 12], Log = new int[1 << 12];

            int n = jobs.Length;
            for (int i = 2; i < (1 << n); i += 1) Log[i] = Log[i >> 1] + 1;
            for (int i = 1; i < (1 << n); i += 1)
                sum[i] = sum[i - (i & -i)] + jobs[Log[i & -i]];
            for (int i = 0; i <= k; i += 1)
                for (int j = 1; j < (1 << n); j += 1) dp[i, j] = 1000000000;
            for (int i = 1; i <= k; i += 1)
                for (int j = 1; j < (1 << n); j += 1)
                {
                    for (int x = j; x == 1; x = (x - 1) & j)
                        dp[i, j] = Math.Min(Math.Max(dp[i - 1, j ^ x], sum[x]), dp[i, j]);
                }
            return dp[k, (1 << n) - 1];
        }


        /// <summary>
        /// 5650. 执行交换操作后的最小汉明距离
        /// Medium
        /// </summary>
        /// <param name="source"></param>
        /// <param name="target"></param>
        /// <param name="allowedSwaps"></param>
        /// <returns></returns>
        public int MinimumHammingDistance(int[] source, int[] target, int[][] allowedSwaps)
        {
            int res = 0;



            return res;
        }


        /// <summary>
        /// 5652. 交换链表中的节点
        /// Medium
        /// </summary>
        /// <param name="head"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public ListNode SwapNodes(ListNode head, int k)
        {
            List<int> list = new List<int>();
            while (head != null)
            {
                list.Add(head.val);
                head = head.next;
            }
            int tmp = 0;
            tmp = list[k - 1];
            list[k - 1] = list[list.Count - k];
            list[list.Count - k] = tmp;

            ListNode dummyHead = new ListNode(-1);
            ListNode preNode = dummyHead;
            for (int i = 0; i < list.Count; i++)
            {
                ListNode curNode = new ListNode(list[i]);
                preNode.next = curNode;
                preNode = preNode.next;
            }
            return dummyHead.next;
        }



        /// <summary>
        /// 5652. 交换链表中的节点
        /// Medium
        /// </summary>
        /// <param name="head"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public ListNode SwapNodes1(ListNode head, int k)
        {
            if (head == null || head.next == null) return head;
            ListNode newHead = new ListNode(-1);
            newHead.next = head;
            ListNode fast = head;
            ListNode slow = head;
            for (int i = 0; i < k - 1; i++)
            {
                fast = fast.next;
            }
            ListNode l = fast;
            fast = fast.next;
            while (fast.next != null)
            {
                fast = fast.next;
                slow = slow.next;
            }
            ListNode r = slow;
            Swap(l, r);
            return newHead.next;

            void Swap(ListNode l1, ListNode l2)
            {
                int tmp = l1.val;
                l1.val = l2.val;
                l2.val = tmp;
            }
        }




        /// <summary>
        /// 5649. 解码异或后的数组
        /// </summary>
        /// <param name="encoded"></param>
        /// <param name="first"></param>
        /// <returns></returns>
        public int[] Decode(int[] encoded, int first)
        {
            int[] res = new int[encoded.Length + 1];
            res[0] = first;
            for (int i = 0; i < encoded.Length; i++)
            {
                res[i + 1] = encoded[i] ^ res[i];
            }
            return res;
        }
        #endregion

        /// <summary>
        /// 39. 组合总和
        /// Medium
        /// </summary>
        /// <param name="candidates"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public IList<IList<int>> CombinationSum(int[] candidates, int target)
        {
            Array.Sort(candidates);
            List<IList<int>> res = new List<IList<int>>();
            int len = candidates.Length;
            List<int> combine = new List<int>();
            DFS(candidates, target, res, combine, 0);
            return res;

            void DFS(int[] candidates, int target, IList<IList<int>> res, IList<int> combine, int idx)
            {
                if (idx == candidates.Length) return;
                if (target == 0)
                {
                    res.Add(new List<int>(combine));
                    return;
                }
                DFS(candidates, target, res, combine, idx + 1);
                if (target - candidates[idx] >= 0)
                {
                    combine.Add(candidates[idx]);
                    DFS(candidates, target - candidates[idx], res, combine, idx);
                    combine.RemoveAt(combine.Count - 1);
                }
            }
        }


        /// <summary>
        /// 6. Z 字形变换
        /// Medium
        /// </summary>
        /// <param name="s"></param>
        /// <param name="numRows"></param>
        /// <returns></returns>
        public string Convert_Z(string s, int numRows)
        {
            if (numRows == 1) return s;
            List<StringBuilder> rows = new List<StringBuilder>();
            for (int i = 0; i < Math.Min(numRows, s.Length); i++)
            {
                rows.Add(new StringBuilder());
            }
            int curRow = 0;
            bool goingDown = false;
            for (int i = 0; i < s.Length; i++)
            {
                rows[curRow].Append(s[i]);
                if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
                curRow += goingDown ? 1 : -1;
            }
            StringBuilder res = new StringBuilder();
            foreach (var item in rows)
            {
                res.Append(item);
            }
            return res.ToString();
        }

        /// <summary>
        /// 258. 各位相加
        /// Easy
        /// </summary>
        /// <param name="num"></param>
        /// <returns></returns>
        public int AddDigits(int num)
        {
            // X = 100 * a + 10 * b + c = 99 * a + 9 * b + (a + b + c);
            return (num - 1) % 9 + 1;
            // 递归
            // if (num < 10) return num;
            // string numStr = num.ToString();
            // int sum = 0;
            // for (int i = 0; i < numStr.Length; i++)
            // {
            //     sum += numStr[i] - '0';
            // }
            // return AddDigits(sum);
        }


        /// <summary>
        /// 237. 删除链表中的节点
        /// Easy
        /// </summary>
        /// <param name="node"></param>
        public void DeleteNode(ListNode node)
        {
            /*
            *  比如 [4, 5, 1, 9] 输入 5
            *  把 5 这里的值设为他下一个的值 1          [4, 1, 1, 9]
            *  然后直接指向 9, 就跳过了原来的 1, 结束   [4, 1, 9]
            */
            node.val = node.next.val;
            node.next = node.next.next;
        }


        /// <summary>
        /// 392. 判断子序列
        /// Easy
        /// </summary>
        /// <param name="s"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public bool IsSubsequence(string s, string t)
        {
            // 双指针
            int i = 0, j = 0;
            while (i < s.Length && j < t.Length)
            {
                if (s[i] == t[j])
                {
                    i++;
                    j++;
                }
                else
                {
                    j++;
                }
            }
            return i == s.Length;

            // 使用栈记录
            // bool res = true;
            // Stack<char> stack = new Stack<char>();
            // for (int i = t.Length - 1; i >= 0; i--)
            // {
            //     stack.Push(t[i]);
            // }
            // for (int i = 0; i < s.Length;)
            // {
            //     if (stack.Count == 0) { return false; }
            //     if (stack.Peek() == s[i])
            //     {
            //         stack.Pop();
            //         i++;
            //     }
            //     else
            //     {
            //         stack.Pop();
            //     }
            // }
            // return res;
        }

        /// <summary>
        /// 374. 猜数字大小
        /// Easy
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        int GuessNumberTarget = 0;
        public int GuessNumber(int n)
        {
            int ans = -1;
            int low = 1, high = n;
            while (low <= high)
            {
                int mid = low + (high - low) / 2;
                int res = guess(mid);
                if (res == 0) return mid;
                else if (res < 0) high = mid - 1;
                else low = mid + 1;
            }
            return ans;

            int guess(int n)
            {
                if (n > GuessNumberTarget) return -1;
                if (n < GuessNumberTarget) return 1;
                return 0;
            }
        }


        /// <summary>
        /// 383. 赎金信
        /// Easy
        /// </summary>
        /// <param name="ransomNote"></param>
        /// <param name="magazine"></param>
        /// <returns></returns>
        public bool CanConstruct(string ransomNote, string magazine)
        {
            var dict = magazine.GroupBy(x => x)
                            .ToDictionary(x => x.Key, x => x.Count());
            for (int i = 0; i < ransomNote.Length; i++)
            {
                if (dict.ContainsKey(ransomNote[i]))
                {
                    dict[ransomNote[i]]--;
                    if (dict[ransomNote[i]] == 0) dict.Remove(ransomNote[i]);
                }
                else
                {
                    return false;
                }
            }
            return true;
        }


        /// <summary>
        /// 190. 颠倒二进制位
        /// Easy
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public uint reverseBits(uint n)
        {
            uint res = 0;
            for (int i = 0; i < 32; i++)
            {
                uint bit = n & 1;
                n = n >> 1;
                res = (res << 1) ^ bit;
            }
            return res;
        }

        /// <summary>
        /// 160. 相交链表
        /// Easy
        /// </summary>
        /// <param name="headA"></param>
        /// <param name="headB"></param>
        /// <returns></returns>
        public ListNode GetIntersectionNode(ListNode headA, ListNode headB)
        {
            /*
                union 为如果存在的相同的部分
                A = a + union
                B = b + union
                
                A + B = B + A
                A(a + union) + b = B(b + union) + a
            */
            if (headA == null || headB == null) return null;
            ListNode a = headA, b = headB;
            while (a != b)
            {
                a = a == null ? headB : a.next;
                b = b == null ? headA : b.next;
            }
            return a;
        }



        /// <summary>
        /// 547. 省份数量
        /// Medium
        /// </summary>
        /// <param name="isConnected"></param>
        /// <returns></returns>
        public int FindCircleNum(int[][] isConnected)
        {
            int n = isConnected.Length;
            UnionFind uf = new UnionFind(n);
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (isConnected[i][j] == 1)
                    {
                        uf.Union(i, j);
                    }
                }
            }
            return uf.size;
            // int res = 0;
            // int len = isConnected.Length;
            // bool[] visited = new bool[len];
            // for (int i = 0; i < len; i++)
            // {
            //     if (!visited[i])
            //     {
            //         DFS(isConnected, visited, len, i);
            //         res++;
            //     }
            // }
            // return res;

            // void DFS(int[][] isConnected, bool[] visited, int len, int i)
            // {
            //     for (int j = 0; j < len; j++)
            //     {
            //         if (isConnected[i][j] == 1 && !visited[j])
            //         {
            //             visited[j] = true;
            //             DFS(isConnected, visited, len, j);
            //         }
            //     }
            // }
        }



        /// <summary>
        /// 990. 等式方程的可满足性
        /// Medium
        /// </summary>
        /// <param name="equations"></param>
        /// <returns></returns>
        public bool EquationsPossible(string[] equations)
        {
            // 并查集
            int[] parents = new int[26];
            for (int i = 0; i < 26; i++)
            {
                parents[i] = i;
            }
            foreach (var item in equations)
            {
                if (item[1] == '=')
                {
                    int idx1 = item[0] - 'a';
                    int idx2 = item[3] - 'a';
                    Union(parents, idx1, idx2);
                }
            }
            foreach (var item in equations)
            {
                if (item[1] == '!')
                {
                    int idx1 = item[0] - 'a';
                    int idx2 = item[3] - 'a';
                    if (Find(parents, idx1) == Find(parents, idx2))
                    {
                        return false;
                    }
                }
            }
            return true;

            int Find(int[] parents, int idx)
            {
                while (parents[idx] != idx)
                {
                    parents[idx] = parents[parents[idx]];
                    idx = parents[idx];
                }
                return idx;
            }

            void Union(int[] parents, int idx1, int idx2)
            {
                parents[Find(parents, idx1)] = Find(parents, idx2);
            }
        }


        /// <summary>
        /// 399. 除法求值
        /// Medium
        /// </summary>
        /// <param name="equations"></param>
        /// <param name="values"></param>
        /// <param name="queries"></param>
        /// <returns></returns>
        public double[] CalcEquation(IList<IList<string>> equations, double[] values, IList<IList<string>> queries)
        {
            double[] res = new double[queries.Count];
            Dictionary<string, double> dict = new Dictionary<string, double>();
            HashSet<string> set = new HashSet<string>();
            // 根据 equations 和 values 构建初始字典. 由 a / b = 2 可知 b / a = 1 / 2 顺带一起加进去
            // 加逗号是为了应对给的例子是 ["a", "aa"] 这种情况，用于区分 "a, aa" 和 "aa, a"
            for (int i = 0; i < values.Length; i++)
            {
                dict[equations[i][0] + ", " + equations[i][1]] = values[i];
                dict[equations[i][1] + ", " + equations[i][0]] = 1 / values[i];
                set.Add(equations[i][0]);
                set.Add(equations[i][1]);
            }
            // Floyd算法, 简单来说就是根据 a / b = 2, b / c = 3 来求 a / c = 6 的过程
            foreach (var k in set)
            {
                foreach (var i in set)
                {
                    foreach (var j in set)
                    {
                        if (dict.ContainsKey(i + ", " + k) && dict.ContainsKey(k + ", " + j))
                        {
                            dict[i + ", " + j] = dict[i + ", " + k] * dict[k + ", " + j];
                        }
                    }
                }
            }
            // 在字典中查找要求的结果
            for (int i = 0; i < queries.Count; i++)
            {
                res[i] = dict.ContainsKey(queries[i][0] + ", " + queries[i][1])
                    ? dict[queries[i][0] + ", " + queries[i][1]]
                    : -1;
            }
            return res;
        }


        /// <summary>
        /// 830. 较大分组的位置
        /// Easy
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public IList<IList<int>> LargeGroupPositions(string s)
        {
            List<IList<int>> res = new List<IList<int>>();
            int l = 0, r = 0, len = s.Length;
            while (r < len)
            {
                while (r < len && s[l] == s[r])
                {
                    r++;
                }
                if (r - l > 2)
                {
                    res.Add(new List<int> { l, r - 1 });
                }
                l = r;
            }
            return res;
        }


        /// <summary>
        /// 509. 斐波那契数
        /// Easy
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public int Fib(int n)
        {
            if (n < 2) return n;
            int[] f = new int[n + 1];
            f[0] = 0;
            f[1] = 1;
            for (int i = 2; i <= n; i++)
            {
                f[i] = f[i - 1] + f[i - 2];
            }
            return f[n];
        }



        /// <summary>
        /// 435. 无重叠区间
        /// Medium
        /// </summary>
        /// <param name="intervals"></param>
        /// <returns></returns>
        public int EraseOverlapIntervals(int[][] intervals)
        {
            int res = 0;
            if (intervals == null || intervals.Length == 0) return res;
            Array.Sort(intervals, (m1, m2) => m1[0].CompareTo(m2[0]));
            int end = intervals[0][1];
            for (int i = 1; i < intervals.Length; i++)
            {
                if (end > intervals[i][0])
                {
                    end = Math.Min(end, intervals[i][1]);
                    res++;
                }
                else
                {
                    end = intervals[i][1];
                }
            }
            return res;
        }


        /// <summary>
        /// 1046. 最后一块石头的重量
        /// Easy
        /// </summary>
        /// <param name="stones"></param>
        /// <returns></returns>
        public int LastStoneWeight(int[] stones)
        {
            // 递归实现
            if (stones.Length == 0) return 0;
            if (stones.Length == 1) return stones[0];
            if (stones.Length == 2) return Math.Abs(stones[1] - stones[0]);
            Array.Sort(stones);
            if (stones[stones.Length - 3] == 0)
            {
                return stones[stones.Length - 1] - stones[stones.Length - 2];
            }
            stones[stones.Length - 1] = stones[stones.Length - 1] - stones[stones.Length - 2];
            stones[stones.Length - 2] = 0;
            return LastStoneWeight(stones);
        }


        /// <summary>
        /// 3. 无重复字符的最长子串
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public int LengthOfLongestSubstring(string s)
        {
            HashSet<char> set = new HashSet<char>();
            int l = 0, r = 0;
            int n = s.Length;
            int count = 0, max = 0;
            while (r < n)
            {
                if (!set.Contains(s[r]))
                {
                    set.Add(s[r]);
                    r++;
                    count++;
                }
                else
                {
                    set.Remove(s[l]);
                    l++;
                    count--;
                }
                max = Math.Max(max, count);
            }
            return max;
        }


        /// <summary>
        /// 1705. 吃苹果的最大数目
        /// </summary>
        /// <param name="apples"></param>
        /// <param name="days"></param>
        /// <returns></returns>
        public int EatenApples(int[] apples, int[] days)
        {
            if (apples.Length == 1) return Math.Min(apples[0], days[0]);
            int n = apples.Length, endDay = 0, zeroDay = 0;
            for (int i = 0; i < n; i++)
            {
                if (apples[i] != 0 && i + days[i] > endDay)
                {
                    endDay = i + days[i];
                }
                if (i >= endDay)
                {
                    zeroDay++;
                }
            }
            return endDay - zeroDay;
        }



        /// <summary>
        /// 1704. 判断字符串的两半是否相似
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public bool HalvesAreAlike(string s)
        {
            char[] arr = { 'a', 'e', 'i', 'o', 'u' };
            int l = 0, r = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if (i < s.Length / 2)
                {
                    if (arr.Contains(char.ToLower(s[i])))
                    {
                        l++;
                    }
                }
                else
                {
                    if (arr.Contains(char.ToLower(s[i])))
                    {
                        r++;
                    }
                }
            }
            return l == r;
        }


        /// <summary>
        /// 330. 按要求补齐数组
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public int MinPatches(int[] nums, int n)
        {
            long curr_range = 0;
            int m = nums.Length;
            int res = 0;
            for (long i = 1, pos = 0; i <= n; i = curr_range + 1)
            {
                if (pos >= m || i < nums[pos])
                {
                    res++;
                    curr_range += i;
                }
                else
                {
                    curr_range += nums[pos];
                    pos++;
                }
            }
            return res;
        }


        /// <summary>
        /// 455. 分发饼干
        /// </summary>
        /// <param name="g"></param>
        /// <param name="s"></param>
        /// <returns></returns>
        public int FindContentChildren(int[] g, int[] s)
        {
            int child = 0, cookie = 0;
            Array.Sort(g);
            Array.Sort(s);
            while (child < g.Length && cookie < s.Length)
            {
                if (g[child] <= s[cookie])
                {
                    child++;
                }
                cookie++;
            }
            return child;
        }


        /// <summary>
        /// 135. 分发糖果
        /// </summary>
        /// <param name="ratings"></param>
        /// <returns></returns>
        public int Candy(int[] ratings)
        {
            if (ratings == null || ratings.Length == 0) return 0;
            int[] nums = new int[ratings.Length];
            nums[0] = 1;
            // 先正序遍历，如果后一位比前一位高分，就给比前一位多1的糖果，否则给1
            for (int i = 1; i < ratings.Length; i++)
            {
                if (ratings[i] > ratings[i - 1])
                {
                    nums[i] = nums[i - 1] + 1;
                }
                else
                {
                    nums[i] = 1;
                }
            }
            // 再倒叙遍历，如果前一位比后一位高分并且得到的糖果小于或等于后一位，就给前一位孩子比后一位孩子多一个糖果
            for (int i = ratings.Length - 2; i >= 0; i--)
            {
                if (ratings[i] > ratings[i + 1] && nums[i] <= nums[i + 1])
                {
                    nums[i] = nums[i + 1] + 1;
                }
            }
            return nums.Sum();
        }


        /// <summary>
        /// 103. 二叉树的锯齿形层序遍历
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public IList<IList<int>> ZigzagLevelOrder(TreeNode root)
        {
            var res = new List<IList<int>>();
            if (root is null) return res;
            var isRight = true;
            var queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                var count = queue.Count;
                var level = new int[count];
                while (count > 0)
                {
                    var node = queue.Dequeue();
                    level[isRight ? (level.Length - count) : (count - 1)] = node.val;
                    count--;
                    if (node.left != null)
                    {
                        queue.Enqueue(node.left);
                    }
                    if (node.right != null)
                    {
                        queue.Enqueue(node.right);
                    }
                }
                res.Add(level.ToList());
                isRight = !isRight;
            }
            return res;
        }


        /// <summary>
        /// 141. 环形链表
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public bool HasCycle(ListNode head)
        {
            ListNode slow = head;
            ListNode fast = head;
            while (fast != null && fast.next != null)
            {
                fast = fast.next.next;
                slow = slow.next;
                if (fast == slow) return true;
            }
            return false;
        }

        /// <summary>
        /// 70. 爬楼梯
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public int ClimbStairs(int n)
        {
            int p = 0, q = 0, r = 1;
            for (int i = 1; i <= n; i++)
            {
                p = q;
                q = r;
                r = p + q;
            }
            return r;
        }


        /// <summary>
        /// 746. 使用最小花费爬楼梯
        /// </summary>
        /// <param name="cost"></param>
        /// <returns></returns>
        public int MinCostClimbingStairs(int[] cost)
        {
            int l = cost.Length;
            int[] dp = new int[l + 1];
            for (int i = 2; i <= l; i++)
            {
                dp[i] = Math.Min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
            }
            return dp[l];
        }


        /// <summary>
        /// 205. 同构字符串
        /// </summary>
        /// <param name="s"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public bool IsIsomorphic(string s, string t)
        {
            if (s.Length != t.Length) return false;
            var dict = new Dictionary<char, char>();
            for (int i = 0; i < s.Length; i++)
            {
                if (dict.ContainsKey(s[i]))
                {
                    if (dict[s[i]] != t[i])
                    {
                        return false;
                    }
                }
                else
                {
                    if (dict.ContainsValue(t[i]))
                    {
                        return false;
                    }
                    else
                    {
                        dict.Add(s[i], t[i]);
                    }
                }
            }
            return true;
        }


        public int MaximalRectangle(char[][] matrix)
        {
            if (matrix.Length == 0 || matrix[0].Length == 0)
            {
                return 0;
            }
            int[] height = new int[matrix[0].Length];//动态规划，确定每一个点的高，然后 逐层实现  柱状图中最大的矩形
            int max = 0;
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    height[j] = matrix[i][j] == '1' ? (height[j] + 1) : 0;
                }
                int tempmax = LargestRectangleArea(height);//构造  柱状图中最大的矩形，逐层实现
                max = Math.Max(max, tempmax);
            }
            return max;


            int LargestRectangleArea(int[] heights)
            {
                int[] ta = new int[heights.Length];
                int[] leftbound = new int[heights.Length];
                int[] rightbound = new int[heights.Length];
                int top = -1;
                //单调栈--左
                for (int i = 0; i < heights.Length; i++)
                {
                    while (top >= 0 && heights[i] <= heights[ta[top]])
                    {
                        ta[top] = 0;
                        top--;
                    }
                    if (top == -1)
                    {
                        leftbound[i] = -1;
                    }
                    else
                    {
                        leftbound[i] = ta[top];
                    }
                    ta[++top] = i;
                }
                //单调栈--右
                top = -1;
                for (int i = heights.Length - 1; i >= 0; i--)
                {
                    while (top >= 0 && heights[i] <= heights[ta[top]])
                    {
                        ta[top] = 0;
                        top--;
                    }
                    if (top == -1)
                    {
                        rightbound[i] = heights.Length;
                    }
                    else
                    {
                        rightbound[i] = ta[top];
                    }
                    ta[++top] = i;
                }
                int max = 0;
                for (int i = 0; i < heights.Length; i++)
                {
                    max = Math.Max(max, heights[i] * (rightbound[i] - leftbound[i] - 1));
                }
                return max;
            }
        }







        /// <summary>
        ///  5631. 跳跃游戏 VI
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public int MaxResult(int[] nums, int k)
        {
            int n = nums.Length;
            int[] q = new int[n];
            int[] f = new int[n];
            int hd = 1, tl = 1;
            f[0] = nums[0];
            q[1] = 0;
            for (int i = 1; i < n; i++)
            {
                while (hd <= tl && q[hd] < i - k) hd++;
                f[i] = f[q[hd]] + nums[i];
                while (hd <= tl && f[i] > f[q[tl]]) tl--;
                q[++tl] = i;
            }
            return f[n - 1];
        }




        /// <summary>
        /// 5630. 删除子数组的最大得分
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int MaximumUniqueSubarray(int[] nums)
        {
            HashSet<int> set = new HashSet<int>();
            int max = 0;
            int res = 0;
            int l = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                if (!set.Contains(nums[i]))
                {
                    set.Add(nums[i]);
                    max += nums[i];
                }
                else
                {
                    while (l < i && set.Contains(nums[i]))
                    {
                        set.Remove(nums[l]);
                        max -= nums[l];
                        l++;
                    }
                    max += nums[i];
                    set.Add(nums[i]);
                }
                res = Math.Max(res, max);
            }
            return res;
        }



        /// <summary>
        /// 
        /// </summary>
        /// <param name="number"></param>
        /// <returns></returns>
        public string ReformatNumber(string number)
        {
            string tmp = "";
            for (int i = 0; i < number.Length; i++)
            {
                if (char.IsDigit(number[i]))
                {
                    tmp += number[i];
                }
            }
            int flag = 0;
            string res = "";
            int target = 3;
            int ff = 0;
            if (tmp.Length % 3 == 1 && tmp.Length > 6)
            {
                ff = tmp.Length - 4;
            }
            if (tmp.Length == 4)
            {
                target = 2;
            }
            for (int i = 0; i < tmp.Length; i++)
            {
                if (ff != 0)
                {
                    if (ff == i)
                    {
                        target = 2;
                        flag = 2;
                    }
                }
                if (flag == target)
                {
                    res += '-';
                    flag = 0;
                }
                res += tmp[i];
                flag++;
            }
            return res;
        }



        /// <summary>
        /// 316. 去除重复字母
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public string RemoveDuplicateLetters(string s)
        {
            Stack<char> stack = new Stack<char>();
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (stack.Contains(c)) continue;
                while (stack.Any() && stack.Peek() > c && s.IndexOf(stack.Peek(), i) != -1)
                {
                    stack.Pop();
                }
                stack.Push(c);
            }
            char[] chars = new char[stack.Count];
            int idx = 0;
            while (stack.Any())
            {
                chars[idx++] = stack.Pop();
            }
            Array.Reverse(chars);
            return new string(chars);
        }

        /// <summary>
        /// 389. 找不同
        /// </summary>
        /// <param name="s"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public char FindTheDifference(string s, string t)
        {
            int res = 0;
            foreach (char c in t)
            {
                res += c;
            }
            foreach (char c in s)
            {
                res -= c;
            }
            return (char)res;
            // 自己的做法
            // var sArr = s.ToCharArray();
            // var tArr = t.ToCharArray();
            // Array.Sort(sArr);
            // Array.Sort(tArr);
            // for(int i = 0; i < tArr.Length; i++){
            //     if(i == sArr.Length) return tArr[i];
            //     if(sArr[i] != tArr[i]) return tArr[i];
            // }
            // return ' ';
        }


        /// <summary>
        /// 188. 买卖股票的最佳时机 IV
        /// </summary>
        /// <param name="k"></param>
        /// <param name="prices"></param>
        /// <returns></returns>
        public int MaxProfitIV(int k, int[] prices)
        {
            /**
            当k大于等于数组长度一半时, 问题退化为贪心问题此时采用 买卖股票的最佳时机 II
            的贪心方法解决可以大幅提升时间性能, 对于其他的k, 可以采用 买卖股票的最佳时机 III
            的方法来解决, 在III中定义了两次买入和卖出时最大收益的变量, 在这里就是k租这样的
            变量, 即问题IV是对问题III的推广, t[i][0]和t[i][1]分别表示第i比交易买入和卖出时
            各自的最大收益
            **/
            if (k < 1) return 0;
            if (k >= prices.Length / 2) return Greedy(prices);
            int[,] t = new int[k, 2];
            for (int i = 0; i < k; i++)
            {
                t[i, 0] = int.MinValue;
            }
            foreach (var p in prices)
            {
                t[0, 0] = Math.Max(t[0, 0], -p);
                t[0, 1] = Math.Max(t[0, 1], t[0, 0] + p);
                for (int i = 1; i < k; i++)
                {
                    t[i, 0] = Math.Max(t[i, 0], t[i - 1, 1] - p);
                    t[i, 1] = Math.Max(t[i, 1], t[i, 0] + p);
                }
            }
            return t[k - 1, 1];

            int Greedy(int[] prices)
            {
                int max = 0;
                for (int i = 1; i < prices.Length; i++)
                {
                    if (prices[i] > prices[i - 1])
                    {
                        max += prices[i] - prices[i - 1];
                    }
                }
                return max;
            }
        }


        /// <summary>
        /// 123. 买卖股票的最佳时机 III
        /// </summary>
        /// <param name="prices"></param>
        /// <returns></returns>
        public int MaxProfitIII(int[] prices)
        {
            int fstBuy = int.MinValue, fstSell = 0;
            int secBuy = int.MinValue, secSell = 0;
            for (int i = 0; i < prices.Length; i++)
            {
                fstBuy = Math.Max(fstBuy, -prices[i]);
                fstSell = Math.Max(fstSell, fstBuy + prices[i]);
                secBuy = Math.Max(secBuy, fstSell - prices[i]);
                secSell = Math.Max(secSell, secBuy + prices[i]);
            }
            return secSell;
        }


        /// <summary>
        /// 122. 买卖股票的最佳时机 II
        /// </summary>
        /// <param name="prices"></param>
        /// <returns></returns>
        public int MaxProfitII(int[] prices)
        {
            int len = prices.Length;
            if (len == 0) return 0;
            int[,] dp = new int[len, 2];
            dp[0, 0] = 0;
            dp[0, 1] = -prices[0];
            for (int i = 1; i < len; i++)
            {
                dp[i, 0] = Math.Max(dp[i - 1, 0], dp[i - 1, 1] + prices[i]);
                dp[i, 1] = Math.Max(dp[i - 1, 0] - prices[i], dp[i - 1, 1]);
            }
            return dp[len - 1, 0];
        }


        /// <summary>
        /// 121. 买卖股票的最佳时机
        /// </summary>
        /// <param name="prices"></param>
        /// <returns></returns>
        public int MaxProfit(int[] prices)
        {
            int n = prices.Length;
            if (n <= 1) return 0;
            int min = prices[0], max = 0;
            for (int i = 1; i < n; i++)
            {
                max = Math.Max(max, prices[i] - min);
                min = Math.Min(min, prices[i]);
            }
            return max;
        }

        /// <summary>
        /// 714. 买卖股票的最佳时机含手续费
        /// </summary>
        /// <param name="prices"></param>
        /// <param name="fee"></param>
        /// <returns></returns>
        public int MaxProfit(int[] prices, int fee)
        {
            int n = prices.Length;
            int[,] dp = new int[n, 2];
            dp[0, 0] = 0;
            dp[0, 1] = -prices[0];
            for (int i = 1; i < n; i++)
            {
                dp[i, 0] = Math.Max(dp[i - 1, 0], dp[i - 1, 1] + prices[i] - fee);
                dp[i, 1] = Math.Max(dp[i - 1, 1], dp[i - 1, 0] - prices[i]);
            }
            return dp[n - 1, 0];
        }


        /// <summary>
        /// 290. 单词规律
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="s"></param>
        /// <returns></returns>
        public bool WordPattern(string pattern, string s)
        {
            var dict = new Dictionary<char, string>();
            var arr = s.Split(' ');
            if (arr.Length != pattern.Length) return false;
            for (int i = 0; i < pattern.Length; i++)
            {
                if (dict.ContainsKey(pattern[i]))
                {
                    if (dict[pattern[i]] == arr[i])
                    {
                        continue;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    if (dict.ContainsValue(arr[i]))
                    {
                        return false;
                    }
                    else
                    {
                        dict.Add(pattern[i], arr[i]);
                    }
                }
            }
            return true;
        }


        /// <summary>
        /// 738. 单调递增的数字
        /// </summary>
        /// <param name="N"></param>
        /// <returns></returns>
        public int MonotoneIncreasingDigits(int N)
        {
            var arr = N.ToString().ToCharArray();
            var len = arr.Length;
            int flag = len;
            for (int i = len - 1; i >= 1; i--)
            {
                if (arr[i] < arr[i - 1])
                {
                    flag = i;
                    arr[i - 1]--;
                }
            }
            for (int i = flag; i < len; i++)
            {
                arr[i] = '9';
            }
            return Convert.ToInt32(new string(arr));


            // int res = N;
            // int i = 1;
            // while (i <= res / 10)
            // {
            //     int n = res / i % 100;
            //     i *= 10;
            //     if (n / 10 > n % 10)
            //     {
            //         res = res / i * i - 1;
            //     }
            // }
            // return res;
        }


        public int LongestValidParentheses(string s)
        {
            return 0;
        }



        /// <summary>
        /// 面试题 16.26. 计算器
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public int Calculate(string s)
        {
            return 0;
        }



        /// <summary>
        /// 47. 全排列 II
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public IList<IList<int>> PermuteUnique(int[] nums)
        {
            var ans = new List<IList<int>>();
            var perm = new List<int>();
            var vis = new bool[nums.Length];
            Array.Sort(nums);
            Backtrack(nums, ans, 0, perm);
            return ans;

            void Backtrack(int[] nums, List<IList<int>> ans, int idx, List<int> perm)
            {
                if (idx == nums.Length)
                {
                    ans.Add(new List<int>(perm));
                    return;
                }
                for (int i = 0; i < nums.Length; i++)
                {
                    if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && !vis[i - 1]))
                    {
                        continue;
                    }
                    perm.Add(nums[i]);
                    vis[i] = true;
                    Backtrack(nums, ans, idx + 1, perm);
                    vis[i] = false;
                    perm.RemoveAt(idx);
                }
            }
        }


        /// <summary>
        /// 46. 全排列
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public IList<IList<int>> Permute(int[] nums)
        {
            IList<IList<int>> res = new List<IList<int>>();
            IList<int> path = new List<int>();
            DFS(path, nums);
            return res;

            void DFS(IList<int> path, int[] nums)
            {
                if (path.Count == nums.Length)
                {
                    res.Add(new List<int>(path));
                    return;
                }
                foreach (var num in nums)
                {
                    if (path.Contains(num))
                        continue;
                    path.Add(num);
                    DFS(path, nums);
                    path.Remove(num);
                }
            }
        }



        /// <summary>
        /// 49. 字母异位词分组
        /// </summary>
        /// <param name="strs"></param>
        /// <returns></returns>
        public IList<IList<string>> GroupAnagrams(string[] strs)
        {
            var dict = new Dictionary<string, IList<string>>();
            for (int i = 0; i < strs.Length; i++)
            {
                var tmp = strs[i].ToCharArray();
                Array.Sort(tmp);
                var key = new string(tmp);
                if (dict.ContainsKey(key))
                {
                    dict[key].Add(strs[i]);
                }
                else
                {
                    dict.Add(key, new List<string>() { strs[i] });
                }
            }
            return new List<IList<string>>(dict.Values);
        }


        /// <summary>
        /// 217. 存在重复元素
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public bool ContainsDuplicate(int[] nums)
        {
            HashSet<int> set = new HashSet<int>();
            for (int i = 0; i < nums.Length; i++)
            {
                if (set.Contains(nums[i]))
                {
                    return true;
                }
                else
                {
                    set.Add(nums[i]);
                }
            }
            return false;
        }



        /// <summary>
        /// 5245. 堆叠长方体的最大高度
        /// </summary>
        /// <param name="cuboids"></param>
        /// <returns></returns>
        public int MaxHeight(int[][] cuboids)
        {
            for (int i = 0; i < cuboids.Length; i++)
            {
                Array.Sort(cuboids[i]);
            }
            Array.Sort(cuboids, (c1, c2) =>
            {
                if (c1[0] != c2[0]) return c1[0].CompareTo(c2[0]);
                else if (c1[1] != c2[1]) return c1[1].CompareTo(c2[1]);
                else return c1[2].CompareTo(c2[2]);
            });
            int[] dp = new int[cuboids.Length];
            int maxHeight = 0;

            for (int i = 0; i < cuboids.Length; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    if (cuboids[j][1] <= cuboids[i][1] && cuboids[j][2] <= cuboids[i][2])
                    {
                        dp[i] = Math.Max(dp[i], dp[j]);
                    }
                }
                dp[i] += cuboids[i][2];
                maxHeight = Math.Max(maxHeight, dp[i]);
            }

            return maxHeight;
        }


        /// <summary>
        /// 5627. 石子游戏 VII
        /// </summary>
        /// <param name="stones"></param>
        /// <returns></returns>
        public int StoneGameVII(int[] stones)
        {
            int n = stones.Length;
            int[] pre = new int[n + 1];
            for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + stones[i];
            int[,] dp = new int[n, n];
            for (int i = n - 1; i >= 0; i--)
            {
                for (int j = i; j < n; j++)
                {
                    if (i == j)
                    {
                        dp[i, j] = 0;
                        continue;
                    }
                    int L = (pre[j + 1] - pre[i + 1]) - dp[i + 1, j];
                    int R = (pre[j] - pre[i]) - dp[i, j - 1];
                    dp[i, j] = Math.Max(L, R);
                }
            }
            return dp[0, n - 1];
        }


        /// <summary>
        /// 5626. 十-二进制数的最少数目
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public int MinPartitions(string n)
        {
            int max = 0;
            for (int i = 0; i < n.Length; i++)
            {
                max = Math.Max(max, n[i] - '0');
            }
            return max;
        }


        /// <summary>
        /// 5625. 比赛中的配对次数
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public int NumberOfMatches(int n)
        {
            int count = 0;
            int curN = 0;
            while (curN != 1)
            {
                if (n % 2 == 1)
                {
                    count += (n - 1) / 2;
                    curN = (n - 1) / 2 + 1;
                }
                else
                {
                    count += n / 2;
                    curN = n / 2;
                }
                n = curN;
            }
            return count;
        }



        /// <summary>
        /// 376. 摆动序列
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int WiggleMaxLength(int[] nums)
        {
            int n = nums.Length;
            if (n < 2) return n;
            int up = 1, down = 1;
            for (int i = 1; i < n; i++)
            {
                if (nums[i] > nums[i - 1])
                {
                    up = down + 1;
                }
                if (nums[i] < nums[i - 1])
                {
                    down = up + 1;
                }
            }
            return Math.Max(up, down);
        }


        /// <summary>
        /// 649. Dota2 参议院
        /// </summary>
        /// <param name="senate"></param>
        /// <returns></returns>
        public string PredictPartyVictory(string senate)
        {
            int n = senate.Length;
            var radiant = new Queue<int>();
            var dire = new Queue<int>();
            for (int i = 0; i < n; i++)
            {
                if (senate[i] == 'R')
                {
                    radiant.Enqueue(i);
                }
                else
                {
                    dire.Enqueue(i);
                }
            }
            while (radiant.Any() && dire.Any())
            {
                int rIndex = radiant.Peek(), dIndex = dire.Peek();
                if (rIndex < dIndex)
                {
                    radiant.Dequeue();
                }
            }
            return "";

            // 循环统计
            // int Rnumber = 0;//R阵营总人数
            // int Dnumber = 0;//D阵营总人数
            // int curBanR = 0;//当前被ban
            // int curBanD = 0;//当前被ban
            // int totalBanR = 0;//被ban总数
            // int totalBanD = 0;//被ban总数
            // char[] chars = senate.ToCharArray();
            // bool flag = true;
            // while (true)
            // {
            //     for (int i = 0; i < chars.Length; i++)
            //     {
            //         char cur = chars[i];
            //         if (cur == 'R')
            //         {
            //             if (flag)
            //                 Rnumber++;
            //             if (curBanR == 0)
            //             {
            //                 curBanD++;
            //                 totalBanD++;
            //                 if (totalBanD == Dnumber && !flag) return "Radiant";
            //             }
            //             else
            //             {
            //                 curBanR--;
            //                 chars[i] = 'r';
            //             }
            //         }
            //         else if (cur == 'D')
            //         {
            //             if (flag)
            //                 Dnumber++;
            //             if (curBanD == 0)
            //             {
            //                 curBanR++;
            //                 totalBanR++;
            //                 if (totalBanR == Rnumber && !flag) return "Dire";
            //             }
            //             else
            //             {
            //                 curBanD--;
            //                 chars[i] = 'd';
            //             }
            //         }
            //     }
            //     flag = false;
            //     if (totalBanD >= Dnumber) return "Radiant";
            //     if (totalBanR >= Rnumber) return "Dire";
            // }
        }


        /// <summary>
        /// 62. 不同路径
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public int UniquePaths(int m, int n)
        {
            // 组合数学
            decimal res = 1;
            for (int x = n, y = 1; y < m; x++, y++)
            {
                res = Math.Floor(res * x / y);
            }
            return (int)res;
            // 动态规划
            // int[][] f = new int[m][];
            // for (int i = 0; i < m; i++)
            // {
            //     f[i] = new int[n];
            //     f[i][0] = 1;
            // }
            // for (int j = 0; j < n; j++)
            // {
            //     f[0][j] = 1;
            // }
            // for (int i = 1; i < m; i++)
            // {
            //     for (int j = 1; j < n; j++)
            //     {
            //         f[i][j] = f[i - 1][j] + f[i][j - 1];
            //     }
            // }
            // return f[m - 1][n - 1];

        }


        /// <summary>
        /// 120. 三角形最小路径和
        /// </summary>
        /// <param name="triangle"></param>
        /// <returns></returns>
        public int MinimumTotal(IList<IList<int>> triangle)
        {
            int minTotal = 0;

            return minTotal;
        }


        /// <summary>
        /// 1267. 统计参与通信的服务器
        /// </summary>
        /// <param name="grid"></param>
        /// <returns></returns>
        public int CountServers(int[][] grid)
        {
            int res = 0;
            int row = grid.Length;
            int col = grid[0].Length;
            int[] rowCount = new int[row];
            int[] colCount = new int[col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    if (grid[i][j] == 1)
                    {
                        rowCount[i]++;
                        colCount[j]++;
                    }
                }
            }

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    if (grid[i][j] == 1 && (rowCount[i] > 1 || colCount[j] > 1))
                    {
                        res++;
                    }
                }
            }

            return res;
        }

        /// <summary>
        /// 84. 柱状图中最大的矩形
        /// </summary>
        /// <param name="heights"></param>
        /// <returns></returns>
        public int LargestRectangleArea(int[] heights)
        {
            // 单调栈
            if (heights == null || heights.Length <= 0) return 0;

            int max = 0;
            var stack = new Stack<int>();
            for (int i = 0; i < heights.Length + 1; ++i)
            {
                while (stack.Any() && ((i == heights.Length) || heights[i] < heights[stack.Peek()]))
                {
                    int height = heights[stack.Pop()];

                    int width = stack.Any() ? i - stack.Peek() - 1 : i;

                    max = Math.Max(max, width * height);
                }
                stack.Push(i);
            }

            return max;

            // 暴力求解，每次左右寻找边界 O(n^2)
            // int res = 0, n = heights.Length;
            // for (int i = 0; i < n; i++)
            // {
            //     int w = 1, h = heights[i], j = i;
            //     while (--j >= 0 && heights[j] >= h)
            //     {
            //         w++;
            //     }
            //     j = i;
            //     while (++j < n && heights[j] >= h)
            //     {
            //         w++;
            //     }
            //     res = Math.Max(res, w * h);
            // }
            // return res;
        }



        /// <summary>
        /// 842. 将数组拆分成斐波那契序列
        /// </summary>
        /// <param name="S"></param>
        /// <returns></returns>
        public IList<int> SplitIntoFibonacci(string S)
        {
            var list = new List<int>();
            Backtrack(list, S, S.Length, 0, 0, 0);
            return list;

            bool Backtrack(List<int> list, string S, int length, int index, int sum, int prev)
            {
                if (index == length) return list.Count >= 3;
                long curLong = 0;
                for (int i = index; i < length; i++)
                {
                    if (i > index && S[index] == '0')
                    {
                        break;
                    }
                    curLong = curLong * 10 + S[i] - '0';
                    if (curLong > int.MaxValue)
                    {
                        break;
                    }
                    int cur = (int)curLong;
                    if (list.Count >= 2)
                    {
                        if (cur < sum)
                        {
                            continue;
                        }
                        else if (cur > sum)
                        {
                            break;
                        }
                    }
                    list.Add(cur);
                    if (Backtrack(list, S, length, i + 1, prev + cur, cur))
                    {
                        return true;
                    }
                    else
                    {
                        list.RemoveAt(list.Count - 1);
                    }
                }
                return false;
            }
        }



        /// <summary>
        /// 861. 翻转矩阵后的得分
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public int MatrixScore(int[][] A)
        {
            int m = A.Length, n = A[0].Length;
            int res = m * (1 << (n - 1));
            for (int j = 1; j < n; j++)
            {
                int nOnes = 0;
                for (int i = 0; i < m; i++)
                {
                    if (A[i][0] == 1)
                    {
                        nOnes += A[i][j];
                    }
                    else
                    {
                        nOnes += (1 - A[i][j]);
                    }
                }
                int k = Math.Max(nOnes, m - nOnes);
                res += k * (1 << (n - j - 1));
            }
            return res;
        }



        /// <summary>
        /// 55. 跳跃游戏
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public bool CanJump(int[] nums)
        {
            int k = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                if (i > k) return false;
                k = Math.Max(k, i + nums[i]);
            }
            return true;
        }


        /// <summary>
        /// 51. N 皇后
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public IList<IList<string>> SolveNQueens(int n)
        {
            var res = new List<IList<string>>();
            var queens = new int[n];
            Array.Fill(queens, -1);
            var col = new HashSet<int>();
            var dias1 = new HashSet<int>();
            var dias2 = new HashSet<int>();
            Backtrack(res, queens, n, 0, col, dias1, dias2);
            return res;

            /// <summary>
            /// 方法一：基于集合的回溯
            /// </summary>
            void Backtrack(IList<IList<string>> solutions, int[] queens, int n, int row
                            , HashSet<int> col, HashSet<int> dias1, HashSet<int> dias2)
            {
                if (row == n)
                {
                    List<string> board = GenerateBoard(queens, n);
                    solutions.Add(board);
                }
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        if (col.Contains(i)) continue;
                        // 方向一的斜线为从左上到右下方向，同一条斜线上的每个位置满足行下标与列下标之差相等
                        int dia1 = row - i;
                        if (dias1.Contains(dia1)) continue;
                        // 方向二的斜线为从右上到左下方向，同一条斜线上的每个位置满足行下标与列下标之和相等
                        int dia2 = row + i;
                        if (dias2.Contains(dia2)) continue;
                        queens[row] = i;
                        col.Add(i);
                        dias1.Add(dia1);
                        dias2.Add(dia2);
                        //下一行递归
                        Backtrack(solutions, queens, n, row + 1, col, dias1, dias2);
                        //回溯
                        queens[row] = -1;
                        col.Remove(i);
                        dias1.Remove(dia1);
                        dias2.Remove(dia2);
                    }
                }
            }

            /// <summary>
            /// 方法二：基于位运算的回溯
            /// </summary>
            void Solve(List<List<string>> res, int[] queens, int n, int row, int col, int dias1, int dias2)
            {
                if (row == n)
                {
                    var board = GenerateBoard(queens, n);
                    res.Add(board);
                }
                else
                {
                    int availablePositions = ((1 << n) - 1) & (~(col | dias1 | dias2));
                    while (availablePositions != 0)
                    {
                        int position = availablePositions & (-availablePositions);
                        availablePositions = availablePositions & (availablePositions - 1);
                        int column = BitCount(position - 1);
                        queens[row] = column;
                        Solve(res, queens, n, row + 1, col | position, (dias1 | position) << 1, (dias2 | position) >> 1);
                        queens[row] = -1;
                    }
                }
            }

            int BitCount(int i)
            {
                int count = 0;
                while (i != 0)
                {
                    count++;
                    i = (i - 1) & i;
                }
                return count;
            }

            List<string> GenerateBoard(int[] queens, int n)
            {
                List<string> board = new List<string>();
                for (int i = 0; i < n; i++)
                {
                    char[] row = new char[n];
                    Array.Fill(row, '.');
                    row[queens[i]] = 'Q';
                    board.Add(new string(row));
                }
                return board;
            }
        }


        /// <summary>
        /// 56. 合并区间
        /// </summary>
        /// <param name="intervals"></param>
        /// <returns></returns>
        public int[][] Merge(int[][] intervals)
        {
            Array.Sort(intervals, (m1, m2) => m1[0].CompareTo(m2[0]));
            List<int[]> res = new List<int[]>();
            int i = 0, n = intervals.Length;
            // while (i < n)
            // {
            //     int l = intervals[i][0];
            //     int r = intervals[i][1];
            //     while (i < n - 1 && r >= intervals[i + 1][0])
            //     {
            //         r = Math.Max(r, intervals[i + 1][1]);
            //         i++;
            //     }
            //     res.Add(new int[] { l, r });
            //     i++;
            // }
            for (i = 0; i < n; i++)
            {
                int l = intervals[i][0];
                int r = intervals[i][1];
                for (; i < n - 1 && r >= intervals[i + 1][0]; i++)
                {
                    r = Math.Max(r, intervals[i + 1][1]);
                }
                res.Add(new int[] { l, r });
            }
            return res.ToArray();
        }


        /// <summary>
        /// 659. 分割数组为连续子序列
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public bool IsPossible(int[] nums)
        {
            // 哈希表 + 贪心
            var dict1 = new Dictionary<int, int>();
            var dict2 = new Dictionary<int, int>();
            dict1 = nums.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            foreach (var num in nums)
            {
                int count = dict1[num];
                if (count > 0)
                {
                    int preCount = dict2.GetValueOrDefault(num - 1, 0);
                    if (preCount > 0)
                    {
                        dict1[num]--;
                        if (dict2.ContainsKey(num - 1))
                        {
                            dict2[num - 1] = preCount - 1;
                        }
                        else
                        {
                            dict2.Add(num - 1, preCount - 1);
                        }
                        if (dict2.ContainsKey(num))
                        {
                            dict2[num] = dict2.GetValueOrDefault(num, 0) + 1;
                        }
                        else
                        {
                            dict2.Add(num, dict2.GetValueOrDefault(num, 0) + 1);
                        }
                    }
                    else
                    {
                        int count1 = dict1.GetValueOrDefault(num + 1, 0);
                        int count2 = dict1.GetValueOrDefault(num + 2, 0);
                        if (count1 > 0 && count2 > 0)
                        {
                            dict1[num] = count - 1;
                            dict1[num + 1] = count1 - 1;
                            dict1[num + 2] = count2 - 1;

                            dict2[num + 2] = dict2.ContainsKey(num + 2) ? dict2[num + 2]++ : 1;
                            if (dict2.ContainsKey(num + 2))
                            {
                                dict2[num + 2]++;
                            }
                            else
                            {
                                dict2[num + 2] = 1;
                            }
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }


        /// <summary>
        /// 1576. 替换所有的问号
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public string ModifyString(string s)
        {
            char[] sb = s.ToCharArray();
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '?')
                {
                    char a = 'a';
                    while ((i > 0 && s[i - 1] == a) || (i < s.Length - 1 && s[i + 1] == a))
                    {
                        a++;
                    }
                    sb[i] = a;
                }
            }
            return new string(sb);
        }


        /// <summary>
        /// 228. 汇总区间
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public IList<string> SummaryRanges(int[] nums)
        {
            List<string> res = new List<string>();
            for (int i = 0, j = 0; j < nums.Length; j++)
            {
                if (j + 1 < nums.Length && nums[j + 1] == nums[j] + 1)
                    continue;
                if (i == j)
                    res.Add(nums[i] + "");
                else
                    res.Add(nums[i] + "->" + nums[j]);
                i = j + 1;
            }
            return res;
        }

        /// <summary>
        /// 151. 翻转字符串里的单词
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public string ReverseWords(string s)
        {
            var arr = s.Trim().Split(' ');
            var res = arr.Where(m => m != "").ToArray();
            Array.Reverse(res);
            return String.Join(" ", res);
        }


        /// <summary>
        /// 1438. 绝对差不超过限制的最长连续子数组
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="limit"></param>
        /// <returns></returns>
        public int LongestSubarray(int[] nums, int limit)
        {
            LinkedList<int> min = new LinkedList<int>();
            LinkedList<int> max = new LinkedList<int>();
            int start = 0, end = 0;
            int res = 0;
            for (; end < nums.Length; end++)
            {
                while (min.Any() && nums[end] < min.Last.Value) min.RemoveLast();
                while (max.Any() && nums[end] > max.Last.Value) max.RemoveLast();
                min.AddLast(nums[end]);
                max.AddLast(nums[end]);
                while (max.First.Value - min.First.Value > limit)
                {
                    if (nums[start] == min.First.Value) min.RemoveFirst();
                    if (nums[start] == max.First.Value) max.RemoveFirst();
                    start++;
                }
                res = Math.Max(res, end - start + 1);
            }
            return res;
        }



        /// <summary>
        /// 42. 接雨水
        /// 面试题 17.21. 直方图的水量
        /// </summary>
        /// <param name="height"></param>
        /// <returns></returns>
        public int Trap(int[] height)
        {
            // 双指针
            // int res = 0;
            // int l = 0;
            // int r = height.Length - 1;
            // int tmp = 0;
            // while (l < r)
            // {
            //     if (height[l] <= height[r])
            //     {
            //         tmp = Math.Max(tmp, height[l]);
            //         res += tmp - height[l++];
            //     }
            //     else
            //     {
            //         tmp = Math.Max(tmp, height[r]);
            //         res += tmp - height[r--];
            //     }
            // }
            // return res;

            if (height.Length <= 2)
            {
                return 0;
            }
            int result = 0;
            int maxIndex = 0;
            for (int i = 0; i < height.Length; i++)
            {
                if (height[i] > height[maxIndex])
                {
                    maxIndex = i;
                }
            }
            int left = 0;
            int start = left;
            while (left < maxIndex)
            {
                if (left < maxIndex)
                {
                    if (height[left] >= height[start])
                    {
                        result += height[left] - height[start];
                        start++;
                    }
                    else
                    {
                        left = start;
                    }
                }
            }

            int right = height.Length - 1;
            start = right;
            while (start > maxIndex)
            {
                if (height[right] >= height[start])
                {
                    result += height[right] - height[start];
                    start--;
                }
                else
                {
                    right = start;
                }
            }
            return result;
        }



        /// <summary>
        /// 1074. 元素和为目标值的子矩阵数量
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public int NumSubmatrixSumTarget(int[][] matrix, int target)
        {
            if (matrix == null || matrix.Length == 0 || matrix[0].Length == 0) return 0;
            // 预处理二维矩阵前缀和
            int m = matrix.Length;
            int n = matrix[0].Length;
            int[][] sum = new int[m + 1][];
            sum[0] = new int[n + 1];
            for (int i = 1; i <= m; i++)
            {
                sum[i] = new int[n + 1];
                for (int j = 1; j <= n; j++)
                {
                    sum[i][j] = sum[i - 1][j] + sum[i][j - 1] + matrix[i - 1][j - 1] - sum[i - 1][j - 1];
                }
            }
            // 枚举所有子矩阵
            int res = 0;
            // x1,y1为左上角， x2,y2为右下角
            for (int x1 = 0; x1 < m; x1++)
            {
                for (int y1 = 0; y1 < n; y1++)
                {
                    for (int x2 = x1; x2 < m; x2++)
                    {
                        for (int y2 = y1; y2 < n; y2++)
                        {
                            int tmp;
                            tmp = sum[x2 + 1][y2 + 1] - sum[x1][y2 + 1] - sum[x2 + 1][y1] + sum[x1][y1];
                            if (tmp == target) res++;
                        }
                    }
                }
            }
            return res;
        }


        /// <summary>
        /// 10. 正则表达式匹配
        /// </summary>
        /// <param name="s"></param>
        /// <param name="p"></param>
        /// <returns></returns>
        // public bool IsMatch(string s, string p)
        // {

        // }

        /// <summary>
        /// 204. 计数质数
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public int CountPrimes(int n)
        {
            // 线性筛
            // List<int> primes = new List<int>();
            // int[] isPrime = new int[n];
            // Array.Fill(isPrime, 1);
            // for (int i = 2; i < n; i++)
            // {
            //     if (isPrime[i] == 1)
            //     {
            //         primes.Add(i);
            //     }
            //     for (int j = 0; j < primes.Count && i * primes[j] < n; j++)
            //     {
            //         isPrime[i * primes[j]] = 0;
            //         if (i % primes[j] == 0) break;
            //     }
            // }
            // return primes.Count;


            // 暴力法(优化后)
            // if (n < 3) return 0;
            // //从3开始验算，所以初始值为1（2为质数）。
            // int count = 1;
            // for (int i = 3; i < n; i++)
            // {
            //     //当某个数为 2 的 n 次方时（n为自然数），其 & (n - 1) 所得值将等价于取余运算所得值
            //     //*如果 x = 2^n ，则 x & (n - 1) == x % n
            //     if ((i & 1) == 0) continue;
            //     bool sign = true;
            //     //用 j * j <= i 代替 j <= √i 会更好。
            //     //因为我们已经排除了所有偶数，所以每次循环加二将规避偶数会减少循环次数
            //     for (int j = 3; j * j <= i; j += 2)
            //     {
            //         if (i % j == 0)
            //         {
            //             sign = false;
            //             break;
            //         }
            //     }
            //     if (sign) count++;
            // }
            // return count;

            // 厄拉多塞筛法
            // 时间复杂度：O(n\log \log n)O(nloglogn)
            // 空间复杂度：O(n)O(n)
            bool[] isPrime = new bool[n];
            Array.Fill(isPrime, true);
            for (int i = 2; i * i < n; i++)
            {
                if (isPrime[i])
                {
                    for (int j = i * i; j < n; j += i)
                    {
                        isPrime[j] = false;
                    }
                }
            }
            int count = 0;
            for (int i = 2; i < n; i++)
            {
                if (isPrime[i]) count++;
            }
            return count;
        }


        /// <summary>
        /// 98. 验证二叉搜索树
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public bool IsValidBST(TreeNode root)
        {
            int? last = null;
            return InOrder(root);

            bool InOrder(TreeNode node)
            {
                if (node is null) return true;
                // 中序遍历
                if (InOrder(node.left))  // 左
                {
                    if (last is null || node.val > last) // 根
                    {
                        last = node.val;
                        return InOrder(node.right); // 右
                    }
                }
                return false;
            }
        }


        /// <summary>
        /// 104. 二叉树的最大深度
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public int MaxDepth(TreeNode root)
        {
            // 广度优先
            var res = 0;
            var queue = new Queue<TreeNode>();
            if (root != null) queue.Enqueue(root);

            while (queue.Any())
            {
                res++;
                var list = queue.ToList();
                queue.Clear();
                foreach (var node in list)
                {
                    if (node.left != null) queue.Enqueue(node.left);
                    if (node.right != null) queue.Enqueue(node.right);
                }
            }
            return res;

            // 深度优先
            // if (root is null) return 0;
            // return Math.Max(MaxDepth(root.left), MaxDepth(root.right)) + 1;
        }



        /// <summary>
        /// 28. 实现 strStr()
        /// </summary>
        /// <param name="haystack"></param>
        /// <param name="needle"></param>
        /// <returns></returns>
        public int StrStr(string haystack, string needle)
        {
            // KMP
            // dp[i] 表示 i 前面的都匹配成功时的相同前后缀长度
            var dp = new int[needle.Length];
            // 填表
            for (int l = 0, r = 1; r < needle.Length - 1;)
            {
                if (needle[l].Equals(needle[r]))
                {
                    dp[++r] = ++l;
                }
                else if (l == 0)
                {
                    dp[++r] = 0;
                }
                else
                {
                    l = dp[l];
                }
            }

            // 字符串匹配
            for (int i = 0, j = 0; i - j <= haystack.Length - needle.Length; i += j == 0 ? 1 : 0, j = dp[j])
            {
                for (; j < needle.Length; j++, i++)
                {
                    if (!haystack[i].Equals(needle[j])) break;
                }
                if (j == needle.Length) return i - needle.Length;
            }

            return -1;
        }


        /// <summary>
        /// 字符串转整数用状态枚举
        /// </summary>
        private enum MyAtoiState
        {
            NONE,
            DIGIT,
            FINISH,
        }

        /// <summary>
        /// 8. 字符串转换整数 (atoi)
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public int MyAtoi(string s)
        {
            // 自动机
            var state = new MyAtoiState();
            var sb = new StringBuilder();
            foreach (var item in s.TrimStart())
            {
                switch (state)
                {
                    case MyAtoiState.NONE:
                        if (item.Equals('-') || item.Equals('+') || char.IsDigit(item))
                        {
                            state = MyAtoiState.DIGIT;
                            sb.Append(item);
                        }
                        else
                        {
                            state = MyAtoiState.FINISH;
                        }
                        break;
                    case MyAtoiState.DIGIT:
                        if (Char.IsDigit(item))
                        {
                            sb.Append(item);
                        }
                        else
                        {
                            state = MyAtoiState.FINISH;
                        }
                        break;
                    default:
                        break;
                }
                if (state.Equals(MyAtoiState.FINISH)) break;
            }
            try
            {
                return int.Parse(sb.ToString());
            }
            catch (OverflowException)
            {
                return sb[0].Equals('-') ? int.MinValue : int.MaxValue;
            }
            catch (FormatException)
            {
                return 0;
            }
        }


        /// <summary>
        /// 387. 字符串中的第一个唯一字符
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public int FirstUniqChar(string s)
        {
            // for (int i = 0; i < s.Length; i++)
            // {
            //     int firstIndex = s.IndexOf(s[i]);
            //     int lastIndex = s.LastIndexOf(s[i]);
            //     if (firstIndex == lastIndex) return i;
            // }
            // return -1;
            var res = s.GroupBy(x => x).Where(y => y.Count() == 1).FirstOrDefault();
            return res is null ? -1 : s.IndexOf(res.First());
        }

        /// <summary>
        /// 344. 反转字符串
        /// </summary>
        /// <param name="s"></param>
        public void ReverseString(char[] s)
        {
            for (int l = 0, r = s.Length - 1; l < r; l++, r--)
            {
                (s[l], s[r]) = (s[r], s[l]);
            }
        }


        /// <summary>
        /// 1. 两数之和
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public int[] TwoSum(int[] nums, int target)
        {
            int len = nums.Length;
            var dict = new Dictionary<int, int>();
            for (int i = 0; i < len; i++)
            {
                if (dict.ContainsKey(target - nums[i]))
                {
                    return new int[] { dict[target - nums[i]], i };
                }
                else
                {
                    dict.Add(nums[i], i);
                }
            }
            return new int[] { 0, 0 };
        }


        /// <summary>
        /// 350. 两个数组的交集 II
        /// </summary>
        /// <param name="nums1"></param>
        /// <param name="nums2"></param>
        /// <returns></returns>
        public int[] Intersect(int[] nums1, int[] nums2)
        {
            // 排序后双指针遍历
            Array.Sort(nums1);
            Array.Sort(nums2);
            int len1 = nums1.Length, len2 = nums2.Length;
            var res = new List<int>();
            int idx1 = 0, idx2 = 0, idx = 0;
            while (idx1 < len1 && idx2 < len2)
            {
                if (nums1[idx1] < nums2[idx2])
                {
                    idx1++;
                }
                else if (nums1[idx1] > nums2[idx2])
                {
                    idx2++;
                }
                else
                {
                    res.Add(nums1[idx1]);
                    idx1++;
                    idx2++;
                    idx++;
                }
            }
            return res.ToArray();

            // 哈希表计数
            // if (nums1.Length > nums2.Length) return Intersect(nums2, nums1);
            // var res = new List<int>();
            // var dict = new Dictionary<int, int>();
            // foreach (var item in nums1)
            // {
            //     dict[item] = dict.ContainsKey(item) ? dict[item] + 1 : 1;
            // }
            // foreach (var item in nums2)
            // {
            //     if (dict.ContainsKey(item))
            //     {
            //         res.Add(item);
            //         if (--dict[item] == 0)
            //         {
            //             dict.Remove(item);
            //         }
            //     }
            // }
            // return res.ToArray();
        }


        /// <summary>
        /// 189. 旋转数组
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="k"></param>
        public void Rotate(int[] nums, int k)
        {
            k %= nums.Length;
            // 这个没错，但是会超时（超时也是错，菜是原罪）
            // for (int i = 0; i < k; i++)
            // {
            //     var temp = nums[nums.Length - 1];
            //     for (int j = nums.Length - 1; j > 0; j--)
            //     {
            //         nums[j] = nums[j - 1];
            //     }
            //     nums[0] = temp;
            // }
            Array.Reverse(nums);
            Array.Reverse(nums, 0, k);
            Array.Reverse(nums, k, nums.Length - k);
        }


        public void Merge(int[] nums1, int m, int[] nums2, int n)
        {
            int len1 = m - 1, len2 = n - 1, len = m + n - 1;
            while (len1 >= 0 && len2 >= 0)
            {
                nums1[len--] = nums1[len1] > nums2[len2]
                            ? nums1[len1--]
                            : nums2[len2--];
            }
            Array.Copy(nums2, nums1, len2 + 1);
        }


        /// <summary>
        /// 239. 滑动窗口最大值
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public int[] MaxSlidingWindow(int[] nums, int k)
        {
            int n = nums.Length;
            if (n * k == 0) return new int[0];
            int[] res = new int[n - k + 1];
            // 这里的双向列表保存的是数组的下标
            LinkedList<int> linkedList = new LinkedList<int>();
            for (int i = 0; i < n; i++)
            {
                if (linkedList.Count != 0 && linkedList.First.Value < (i - k + 1))
                {
                    //超出窗口长度时删除队首
                    linkedList.RemoveFirst();
                }
                while (linkedList.Count != 0 && nums[i] >= nums[linkedList.Last.Value])
                {
                    //如果遇到大于队尾元素的数就删除队尾，有几个删几个
                    linkedList.RemoveLast();
                }
                linkedList.AddLast(i);
                if (i >= k - 1)
                {
                    res[i - k + 1] = nums[linkedList.First.Value];
                }
            }
            return res;


            // int n = nums.Length;
            // if (n * k == 0) return new int[0];
            // List<int> res = new List<int>();
            // 这里的双向列表保存的是数组值
            // LinkedList<int> linkedList = new LinkedList<int>();
            // for (int i = 0; i < n; i++)
            // {
            //     if (i < k - 1)
            //     {
            //         AddElement(linkedList, nums[i]);
            //         continue;
            //     }
            //     AddElement(linkedList, nums[i]);
            //     res.Add(linkedList.First.Value);
            //     RemoveElement(linkedList, nums[i - k + 1]);
            // }
            // return res.ToArray();

            // void AddElement(LinkedList<int> linkedList, int val)
            // {
            //     while (linkedList.Count != 0 && linkedList.Last.Value < val)
            //     {
            //         linkedList.RemoveLast();
            //     }
            //     linkedList.AddLast(val);
            // }

            // void RemoveElement(LinkedList<int> linkedList, int val)
            // {
            //     if (linkedList.Count != 0 && linkedList.First.Value == val)
            //     {
            //         linkedList.RemoveFirst();
            //     }
            // }
        }


        /// <summary>
        /// 321. 拼接最大数
        /// </summary>
        /// <param name="nums1"></param>
        /// <param name="nums2"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public int[] MaxNumber(int[] nums1, int[] nums2, int k)
        {
            int m = nums1.Length, n = nums2.Length;
            int[] res = new int[k];
            for (int i = Math.Max(0, k - n); i <= k && i <= m; i++)
            {
                int[] arr = Merge(MaxArr(nums1, i), MaxArr(nums2, k - i), k);
                if (Compare(arr, 0, res, 0)) res = arr;
            }
            return res;

            int[] MaxArr(int[] nums, int k)
            {
                int[] res = new int[k];
                int n = nums.Length;
                for (int i = 0, j = 0; i < n; i++)
                {
                    while (n - i + j > k && j > 0 && nums[i] > res[j - 1]) j--;
                    if (j < k) res[j++] = nums[i];
                }
                return res;
            }

            int[] Merge(int[] nums1, int[] nums2, int k)
            {
                int[] res = new int[k];
                for (int i = 0, j = 0, r = 0; r < k; r++)
                    res[r] = Compare(nums1, i, nums2, j) ? nums1[i++] : nums2[j++];
                return res;
            }

            bool Compare(int[] nums1, int i, int[] nums2, int j)
            {
                while (i < nums1.Length && j < nums2.Length && nums1[i] == nums2[j])
                {
                    i++;
                    j++;
                }
                return j == nums2.Length || (i < nums1.Length && nums1[i] > nums2[j]);
            }
        }


        /// <summary>
        /// 1519. 子树中标签相同的节点数
        /// </summary>
        /// <param name="n"></param>
        /// <param name="edges"></param>
        /// <param name="labels"></param>
        /// <returns></returns>
        // public int[] CountSubTrees(int n, int[][] edges, string labels)
        // {
        //     Dictionary<int, List<int>> dict = new Dictionary<int, List<int>>();
        //     foreach (var edge in edges)
        //     {
        //         int node0 = edge[0], node1 = edge[1];
        //         List<int> list0 = dict.GetValueOrDefault(node0, new List<int>());
        //         List<int> list1 = dict.GetValueOrDefault(node1, new List<int>());
        //         list0.Add(node0);
        //         list1.Add(node1);
        //         dict.Add(node0, list0);
        //         dict.Add(node1, list1);
        //     }

        // }


        /// <summary>
        /// 37. 解数独
        /// </summary>
        /// <param name="board"></param>
        public void SolveSudoku(char[][] board)
        {
            bool[][] row = new bool[9][];
            bool[][] col = new bool[9][];
            bool[][] block = new bool[9][];

            for (int i = 0; i < 9; i++)
            {
                row[i] = new bool[9];
                col[i] = new bool[9];
                block[i] = new bool[9];
                for (int j = 0; j < 9; j++)
                {
                    row[i][j] = false;
                    col[i][j] = false;
                    block[i][j] = false;
                }
            }

            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    if (board[i][j] != '.')
                    {
                        int num = board[i][j] - '1';
                        row[i][num] = true;
                        col[j][num] = true;
                        block[i / 3 * 3 + j / 3][num] = true;
                        Console.WriteLine("INIT BOARD");
                        PrintBoard(board);
                    }
                }
            }

            DFS(board, row, col, block, 0, 0);

            bool DFS(char[][] board, bool[][] row, bool[][] col, bool[][] block, int i, int j)
            {
                // 寻找空位置
                while (board[i][j] != '.')
                {
                    if (++j >= 9)
                    {
                        i++;
                        j = 0;
                    }
                    if (i >= 9)
                    {
                        return true;
                    }
                }
                for (int num = 0; num < 9; num++)
                {
                    int blockIndex = i / 3 * 3 + j / 3;
                    if (!row[i][num] && !col[j][num] && !block[blockIndex][num])
                    {
                        // 递归
                        board[i][j] = (char)('1' + num);
                        row[i][num] = true;
                        col[j][num] = true;
                        block[blockIndex][num] = true;
                        Console.WriteLine("ADD");
                        PrintBoard(board);
                        if (DFS(board, row, col, block, i, j))
                        {
                            return true;
                        }
                        else
                        {
                            // 回溯
                            row[i][num] = false;
                            col[j][num] = false;
                            block[blockIndex][num] = false;
                            board[i][j] = '.';
                            Console.WriteLine("ROLLBACK");
                            PrintBoard(board);
                        }
                    }
                }
                return false;
            }

            void PrintBoard(char[][] board)
            {
                for (int i = 0; i < 9; i++)
                {
                    for (int j = 0; j < 9; j++)
                    {
                        Console.Write(board[i][j] + " ");
                    }
                    Console.WriteLine();
                }
            }
        }

        /// <summary>
        /// 36. 有效的数独
        /// </summary>
        /// <param name="board"></param>
        /// <returns></returns>
        public bool IsValidSudoku(char[][] board)
        {
            Dictionary<char, int>[] row = new Dictionary<char, int>[9];
            Dictionary<char, int>[] col = new Dictionary<char, int>[9];
            Dictionary<char, int>[] box = new Dictionary<char, int>[9];
            for (int i = 0; i < 9; i++)
            {
                row[i] = new Dictionary<char, int>();
                col[i] = new Dictionary<char, int>();
                box[i] = new Dictionary<char, int>();
            }
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    var cur = board[i][j];
                    if (cur.Equals('.'))
                    {
                        continue;
                    }
                    //计算box的索引
                    int k = i / 3 * 3 + j / 3;
                    if (!row[i].ContainsKey(cur) && !col[j].ContainsKey(cur) && !box[k].ContainsKey(cur))
                    {
                        row[i].Add(cur, i);
                        col[j].Add(cur, j);
                        box[k].Add(cur, k);
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            return true;
        }


        /// <summary>
        /// 34. 在排序数组中查找元素的第一个和最后一个位置
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public int[] SearchRange(int[] nums, int target)
        {
            // 系统函数
            // var a = nums.ToList().FindIndex(x => x == target);
            // var b = nums.ToList().FindLastIndex(x => x == target);
            // return new int[] { a, b };

            // 用两次二分分别逼近左右
            int[] res = new int[2];
            res[0] = res[1] = -1;
            int len = nums.Length;
            int l = 0, r = len - 1, mid = 0;
            if (len == 0) return res;
            if (len == 1 && nums[0] != target) return res;
            if (target > nums[r] || target < nums[l]) return res;
            while (l < r)
            {
                mid = (l + r) / 2;
                if (nums[mid] >= target)
                {
                    r = mid;
                }
                else
                {
                    l = mid + 1;
                }
            }
            if (nums[l] != target) return res;
            res[0] = l;
            r = len;
            while (l < r)
            {
                mid = (l + r) / 2;
                if (nums[mid] <= target)
                {
                    l = mid + 1;
                }
                else
                {
                    r = mid;
                }
            }
            res[1] = l - 1;
            return res;
        }


        /// <summary>
        /// 998. 最大二叉树 II
        /// </summary>
        /// <param name="root"></param>
        /// <param name="val"></param>
        /// <returns></returns>
        public TreeNode InsertIntoMaxTree(TreeNode root, int val)
        {
            if (root == null)
            {
                TreeNode pre = new TreeNode(val);
                return pre;
            }
            if (val > root.val)
            {
                TreeNode pre = new TreeNode(val);
                pre.left = root;
                return pre;
            }
            root.right = InsertIntoMaxTree(root.right, val);
            return root;
        }


        /// <summary>
        /// 654. 最大二叉树
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public TreeNode ConstructMaximumBinaryTree(int[] nums)
        {
            return MaxTree(nums, 0, nums.Length - 1);

            TreeNode MaxTree(int[] nums, int l, int r)
            {
                if (l > r) return null;
                int maxIndex = FindMaxIndex(nums, l, r);
                TreeNode root = new TreeNode(nums[maxIndex]);
                root.left = MaxTree(nums, l, maxIndex - 1);
                root.right = MaxTree(nums, maxIndex + 1, r);
                return root;
            }

            int FindMaxIndex(int[] nums, int l, int r)
            {
                int max = nums[l], maxIndex = l;
                for (int i = l; i < r + 1; i++)
                {
                    if (max < nums[i])
                    {
                        max = nums[i];
                        maxIndex = i;
                    }
                }
                return maxIndex;
            }
        }


        /// <summary>
        /// 剑指 Offer 32 - II. 从上到下打印二叉树 II
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public IList<IList<int>> LevelOrder(TreeNode root)
        {
            // 使用队列进行层序遍历
            List<IList<int>> list = new List<IList<int>>();
            if (root == null) return list;
            Queue<TreeNode> queue = new Queue<TreeNode>();
            queue.Enqueue(root);
            while (queue.Count != 0)
            {
                int count = queue.Count;
                List<int> tmp = new List<int>();
                for (int i = 0; i < count; i++)
                {
                    root = queue.Dequeue();
                    tmp.Add(root.val);
                    if (root.left != null)
                    {
                        queue.Enqueue(root.left);
                    }
                    if (root.right != null)
                    {
                        queue.Enqueue(root.right);
                    }
                }
                list.Add(tmp);
            }
            return list;
        }

        // List<IList<int>> list = new List<IList<int>>();
        // //递归进行层序遍历
        // public IList<IList<int>> LevelOrder(TreeNode root)
        // {
        //     lo(root, 0);
        //     return list;

        //     void lo(TreeNode root, int k)
        //     {
        //         if (root != null)
        //         {
        //             if (list.Count <= k)
        //             {
        //                 list.Add(new List<int>());
        //             }
        //             list[k].Add(root.val);
        //             lo(root.left, k + 1);
        //             lo(root.right, k + 1);
        //         }
        //     }
        // }

        /// <summary>
        /// 面试题 04.03. 特定深度节点链表
        /// </summary>
        /// <param name="tree"></param>
        /// <returns></returns>
        public ListNode[] ListOfDepth(TreeNode tree)
        {
            // 使用队列进行层序遍历
            if (tree == null) return new ListNode[0];
            Queue<TreeNode> queue = new Queue<TreeNode>();
            queue.Enqueue(tree);
            List<ListNode> list = new List<ListNode>();
            ListNode dummyHead = new ListNode(-1);
            while (queue.Count != 0)
            {
                ListNode cur = dummyHead;
                int size = queue.Count;
                for (int i = 0; i < size; i++)
                {
                    tree = queue.Dequeue();
                    cur.next = new ListNode(tree.val);
                    cur = cur.next;
                    if (tree.left != null)
                    {
                        queue.Enqueue(tree.left);
                    }
                    if (tree.right != null)
                    {
                        queue.Enqueue(tree.right);
                    }
                }
                list.Add(dummyHead.next);
            }
            return list.ToArray();
        }


        /// <summary>
        /// 589. N叉树的前序遍历
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public IList<int> Preorder(Node root)
        {
            // 非递归实现
            var res = new List<int>();
            var stack = new Stack<Node>();
            if (root == null) return res;
            stack.Push(root);
            while (stack.Count != 0)
            {
                Node node = stack.Pop();
                res.Add(node.val);
                for (int i = node.children.Count - 1; i >= 0; i--)
                {
                    stack.Push(node.children[i]);
                }
            }
            return res;
        }


        /// <summary>
        /// 767. 重构字符串
        /// </summary>
        /// <param name="S"></param>
        /// <returns></returns>
        public string ReorganizeString(string S)
        {
            var mid = (S.Length + 1) / 2;
            var dict = S.GroupBy(x => x)
                        .OrderByDescending(x => x.Count())
                        .ToDictionary(x => x.Key, x => x.Count());
            if (dict.Values.All(x => x <= mid))
            {
                var arr = new char[S.Length];
                Action<int> action = (i) =>
                {
                    var key = dict.Keys.First();
                    arr[i] = key;
                    dict[key]--;
                    if (dict[key] <= 0) dict.Remove(key);
                };
                for (int i = 0; i < arr.Length; i += 2) action(i);
                for (int i = 1; i < arr.Length; i += 2) action(i);
                return arr.ToString();
            }
            return "";

            // int len = S.Length;
            // if (len < 2)
            // {
            //     return S;
            // }
            // int[] counts = new int[26];
            // int maxCount = 0;
            // for (int i = 0; i < len; i++)
            // {
            //     char c = S[i];
            //     counts[c - 'a']++;
            //     maxCount = Math.Max(maxCount, counts[c - 'a']);
            // }
            // if (maxCount > (len + 1) / 2)
            // {
            //     return "";
            // }
        }

        /// <summary>
        /// 493. 翻转对
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int ReversePairs(int[] nums)
        {
            return FindP(0, nums.Length - 1);

            int FindP(int left, int right)
            {
                if (right - left <= 0) return 0;
                var mid = (left + right) / 2;
                var res = FindP(left, mid) + FindP(mid + 1, right);

                for (int i = left, j = mid + 1; i <= mid; i++)
                {
                    while (j <= right && nums[i] > (long)nums[j] * 2) j++;
                    res += j - (mid + 1);
                }

                var temp = new int[right - left + 1];
                for (int i = 0, p1 = left, p2 = mid + 1; i < temp.Length; i++)
                {
                    if (nums[p1] < nums[p2])
                    {
                        temp[i] = nums[p1++];
                        if (p1 > mid)
                        {
                            while (++i < temp.Length) temp[i] = nums[p2++];
                        }
                    }
                    else
                    {
                        temp[i] = nums[p2++];
                        if (p2 > right)
                        {
                            while (++i < temp.Length) temp[i] = nums[p1++];
                        }
                    }
                }
                for (int i = 0; i < temp.Length; i++) nums[left + i] = temp[i];
                return res;
            }
        }


        public int MaximumGap(int[] nums)
        {
            // 基于桶的算法
            int n = nums.Length;
            if (n < 2) return 0;
            int min = nums.Min();
            int max = nums.Max();
            int d = Math.Max(1, (max - min) / (n - 1));
            int bucketSize = (max - min) / d + 1;
            int[][] bucket = new int[bucketSize][];
            int[,] dd = new int[2, 3];
            for (int i = 0; i <= n; i++)
            {
                bucket[i] = new int[2];
                Array.Fill(bucket[i], -1);
            }
            for (int i = 0; i < n; i++)
            {
                int idx = (nums[i] - min) / d;
                if (bucket[idx][0] == -1)
                {
                    bucket[idx][0] = bucket[idx][1] = nums[i];
                }
                else
                {
                    bucket[idx][0] = Math.Min(bucket[idx][0], nums[i]);
                    bucket[idx][1] = Math.Max(bucket[idx][1], nums[i]);
                }
            }

            int ret = 0;
            int prev = -1;
            for (int i = 0; i < bucketSize; i++)
            {
                if (bucket[i][0] == -1)
                {
                    continue;
                }
                if (prev != -1)
                {
                    ret = Math.Max(ret, bucket[i][0] - bucket[prev][1]);
                }
                prev = i;
            }
            return ret;

            // 基数排序
            // int len = nums.Length;
            // if (len < 2) return 0;
            // long exp = 1;
            // int[] buf = new int[len];
            // int maxVal = nums.Max();
            // while (maxVal >= exp)
            // {
            //     int[] cnt = new int[10];
            //     for (int i = 0; i < len; i++)
            //     {
            //         int digit = (nums[i] / (int)exp) % 10;
            //         cnt[digit]++;
            //     }
            //     for (int i = 1; i < 10; i++)
            //     {
            //         cnt[i] += cnt[i - 1];
            //     }
            //     for (int i = len - 1; i >= 0; i--)
            //     {
            //         int digit = (nums[i] / (int)exp) % 10;
            //         buf[cnt[digit] - 1] = nums[i];
            //         cnt[digit]--;
            //     }
            //     Array.Copy(buf, nums, len);
            //     exp *= 10;
            // }

            // int res = 0;
            // for (int i = 1; i < len; i++)
            // {
            //     res = Math.Max(res, nums[i] - nums[i - 1]);
            // }

            // return res;
        }


        public bool ContainsNearbyDuplicate(int[] nums, int k)
        {
            Dictionary<int, int> d = new Dictionary<int, int>();
            int l = 0;
            for (int r = 0; r < nums.Length; ++r)
            {
                if (r - l > k)
                {
                    d.Remove(nums[l]);
                    ++l;
                }
                if (d.ContainsKey(nums[r]))
                    return true;
                d.Add(nums[r], r);
            }
            return false;
        }


        /// <summary>
        /// 1588. 所有奇数长度子数组的和
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        public int SumOddLengthSubarrays(int[] arr)
        {
            //找规律，每个数字被用到的次数
            int res = 0;
            int len = arr.Length;
            for (int i = 0; i < len; i++)
            {
                int left = i + 1, right = len - i,
                left_even = (left + 1) / 2, right_even = (right + 1) / 2,
                left_odd = left / 2, right_odd = right / 2;
                res += (left_even * right_even + left_odd * right_odd) * arr[i];
            }
            return res;

            //前缀和
            // int res = 0;
            // int[] preSum = new int[arr.Length + 1];
            // preSum[0] = arr[0];
            // for (int i = 0; i < arr.Length; i++)
            // {
            //     preSum[i + 1] = preSum[i] + arr[i];
            // }
            // for (int window = 1; window <= arr.Length; window += 2)
            // {
            //     for (int L = 0, R = L + window; R <= arr.Length; L++, R++)
            //     {
            //         res += preSum[R] - preSum[L];
            //     }
            // }
            // return res;
            //暴力模拟
            // int res = 0;
            // for (int window = 1; window <= arr.Length; window += 2)
            // {
            //     for (int L = 0, R = L + window; R <= arr.Length; L++, R++)
            //     {
            //         for (int i = L; i < R; i++)
            //         {
            //             res += arr[i];
            //         }
            //     }
            // }
            // return res;
        }


        /// <summary>
        /// 977. 有序数组的平方
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public int[] SortedSquares(int[] A)
        {
            //双指针从后往前添加
            int start = 0;
            int end = A.Length;
            int i = end - 1;
            int[] nums = new int[end--];
            while (i >= 0)
            {
                nums[i--] = A[start] * A[start] >= A[end] * A[end]
                            ? A[start] * A[start++]
                            : A[end] * A[end--];
            }
            return nums;
            //平方后排序
            // for (int i = 0; i < A.Length; i++)
            // {
            //     A[i] = Math.Abs(A[i] * A[i]);
            // }
            // Array.Sort(A);
            // return A;
        }


        /// <summary>
        /// 410. 分割数组的最大值
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="m"></param>
        /// <returns></returns>
        public int SplitArray(int[] nums, int m)
        {
            int n = nums.Length;
            int[][] f = new int[n + 1][];
            for (int i = 0; i <= n; i++)
            {
                f[i] = new int[m + 1];
                Array.Fill(f[i], int.MaxValue);
            }
            int[] sub = new int[n + 1];
            for (int i = 0; i < n; i++)
            {
                sub[i + 1] = sub[i] + nums[i];
            }
            f[0][0] = 0;
            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= Math.Min(i, m); j++)
                {
                    for (int k = 0; k < i; k++)
                    {
                        f[i][j] = Math.Min(f[i][j], Math.Max(f[k][j - 1], sub[i] - sub[k]));
                    }
                }
            }
            return f[n][m];
        }


        /// <summary>
        /// 1552. 两球之间的磁力
        /// </summary>
        /// <param name="position"></param>
        /// <param name="m"></param>
        /// <returns></returns>
        public int MaxDistance(int[] position, int m)
        {
            Array.Sort(position);
            int left = 1, right = position[position.Length - 1] - position[0], ans = -1;
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (MaxDistanceCheck(mid, position, m))
                {
                    ans = mid;
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }
            return ans;

            bool MaxDistanceCheck(int x, int[] position, int m)
            {
                int pre = position[0], cnt = 1;
                for (int i = 1; i < position.Length; i++)
                {
                    if (position[i] - pre >= x)
                    {
                        pre = position[i];
                        cnt++;
                    }
                }
                return cnt >= m;
            }
        }




        /// <summary>
        /// 1535. 找出数组游戏的赢家
        /// </summary>
        /// <param name="arr"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public int GetWinner(int[] arr, int k)
        {
            int res = Math.Max(arr[0], arr[1]);
            for (int i = 2, count = 1; i < arr.Length; i++)
            {
                if (count == k)
                {
                    return res;
                }
                if (arr[i] > res)
                {
                    count = 1;
                    res = arr[i];
                }
                else
                {
                    count++;
                }
            }
            return res;
        }


        /// <summary>
        /// 面试题 17.10. 主要元素
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int MajorityElement(int[] nums)
        {
            int len = nums.Length;
            if (len == 1) return nums[0];
            Dictionary<int, int> dict = new Dictionary<int, int>();
            for (int i = 0; i < len; i++)
            {
                if (dict.ContainsKey(nums[i]))
                {
                    dict[nums[i]]++;
                    if (dict[nums[i]] > len / 2) return nums[i];
                }
                else
                {
                    dict[nums[i]] = 1;
                }
            }
            return -1;
        }



        /// <summary>
        /// 222. 完全二叉树的节点个数
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public int CountNodes(TreeNode root)
        {
            if (root == null) return 0;
            int ld = GetDepth(root.left);
            int rd = GetDepth(root.right);
            if (ld == rd) return (1 << ld) + CountNodes(root.right);
            else return (1 << rd) + CountNodes(root.left);
        }

        /// <summary>
        /// 获取二叉树某节点的深度
        /// </summary>
        /// <param name="r"></param>
        /// <returns></returns>
        private int GetDepth(TreeNode r)
        {
            int depth = 0;
            while (r != null)
            {
                depth++;
                r = r.left;
            }
            return depth;
        }


        /// <summary>
        /// 402. 移掉K位数字
        /// </summary>
        /// <param name="num"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public string RemoveKdigits(string num, int k)
        {
            var list = new LinkedList<char>();
            for (int i = 0; i < num.Length; i++)
            {
                while (list.Any() && k > 0 && list.Last.Value > num[i])
                {
                    list.RemoveLast();
                    k--;
                }
                list.AddLast(num[i]);
            }
            while (k-- > 0) list.RemoveLast();
            var res = new string(list.ToArray()).TrimStart('0');
            return res.Equals(string.Empty) ? "0" : res;
        }


        /// <summary>
        /// 406. 根据身高重建队列
        /// </summary>
        /// <param name="people"></param>
        /// <returns></returns>
        public int[][] ReconstructQueue(int[][] people)
        {
            Array.Sort(people, (p1, p2) =>
                            p1[0] == p2[0]
                            ? p1[1].CompareTo(p2[1])
                            : p2[0].CompareTo(p1[0]));
            List<int[]> res = new List<int[]>();
            foreach (int[] i in people)
            {
                res.Insert(i[1], i);
            }
            return res.ToArray();
        }



        /// <summary>
        /// 1030. 距离顺序排列矩阵单元格
        /// </summary>
        /// <param name="R"></param>
        /// <param name="C"></param>
        /// <param name="r0"></param>
        /// <param name="c0"></param>
        /// <returns></returns>
        public int[][] AllCellsDistOrder(int R, int C, int r0, int c0)
        {
            int[][] res = new int[R * C][];
            for (int i = 0; i < R; i++)
            {
                for (int j = 0; j < C; j++)
                {
                    int t = i * C + j;
                    res[t] = new int[2];
                    res[t][0] = i;
                    res[t][1] = j;
                }
            }
            Array.Sort(res, (a1, a2) => (Math.Abs(a1[0] - r0) + Math.Abs(a1[1] - c0))
                                        .CompareTo(Math.Abs(a2[0] - r0) + Math.Abs(a2[1] - c0)));

            return res;
        }





        /// <summary>
        /// 148. 排序链表
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public ListNode SortList(ListNode head)
        {
            ListNodeHelper lnHelper = new ListNodeHelper();
            ListNode dummyHead = new ListNode(0);
            dummyHead.next = head;
            ListNode p = head;
            int length = 0;
            while (p != null)
            {
                length++;
                p = p.next;
            }
            for (int size = 1; size < length; size <<= 1)
            {
                ListNode cur = dummyHead.next;
                ListNode tail = dummyHead;
                while (cur != null)
                {
                    ListNode left = cur;
                    ListNode right = lnHelper.Cut(left, size);
                    cur = lnHelper.Cut(right, size);
                    tail.next = lnHelper.Merge(left, right);
                    while (tail.next != null)
                    {
                        tail = tail.next;
                    }
                }
            }
            return dummyHead.next;
        }


        /// <summary>
        /// 147. 对链表进行插入排序
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public ListNode InsertionSortList(ListNode head)
        {
            // dummy为虚拟头节点，一直指向head
            // pre为已排序的部分的首节点
            ListNode dummy = new ListNode(0), pre;
            dummy.next = head;

            while (head != null && head.next != null)
            {
                //若已排序就继续指向下一个
                if (head.val <= head.next.val)
                {
                    head = head.next;
                    continue;
                }
                //若后一个比前一个大，需要开始在前面已排序节点中找要插入的位置
                //所以要初始化pre，重新指向首节点
                pre = dummy;
                //在已排序部分一个一个比下去，找到要插入的位置
                while (pre.next.val < head.next.val)
                {
                    pre = pre.next;
                }
                //找到位置就插入进去（就是把后面的接到前面去）
                ListNode curr = head.next;
                head.next = curr.next;
                curr.next = pre.next;
                pre.next = curr;
            }
            //循环结束返回虚拟头节点指向的next就是要的结果
            //因为dummy一直就指向头节点
            return dummy.next;
        }

        public int[][] Transpose(int[][] A)
        {
            int[][] B = new int[A[0].Length][];
            for (int i = 0; i < A[0].Length; i++)
            {
                B[i] = new int[A.Length];
                for (int j = 0; j < A.Length; j++)
                {
                    B[i][j] = A[j][i];
                }
            }
            return B;
        }


        /// <summary>
        /// 452. 用最少数量的箭引爆气球
        /// </summary>
        /// <param name="points"></param>
        /// <returns></returns>
        public int FindMinArrowShots(int[][] points)
        {
            if (points.Length < 1)
            {
                return 0;
            }
            Array.Sort(points, (x1, x2) => x1[1].CompareTo(x2[1]));
            int temp = points[0][1];
            int result = 1;
            foreach (var point in points)
            {
                if (point[0] > temp)
                {
                    temp = point[1];
                    result++;
                }
            }
            return result;
        }

        public int Massage(int[] nums)
        {
            int len = nums.Length;
            if (len == 0)
            {
                return 0;
            }
            if (len == 1)
            {
                return nums[0];
            }
            int[] dp = new int[len];
            dp[0] = nums[0];
            dp[1] = Math.Max(nums[0], nums[1]);
            for (int i = 2; i < len; i++)
            {
                dp[i] = Math.Max(dp[i - 1], dp[i - 2] + nums[i]);
            }
            return dp[len - 1];
        }

        public void MoveZeroes(int[] nums)
        {
            ShowList(nums);
            int j = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] != 0)
                {
                    var tmp = nums[i];
                    nums[i] = nums[j];
                    nums[j] = tmp;
                    Console.WriteLine($"j = {j}, i = {i}");
                    j++;
                    ShowList(nums);
                }
            }
        }

        public void ShowList(int[] nums)
        {
            string result = "";
            foreach (var num in nums)
            {
                result += num + " ";
            }
            Console.WriteLine(result);
        }

    }
}