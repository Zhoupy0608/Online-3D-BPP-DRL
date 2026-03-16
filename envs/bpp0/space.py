import numpy as np
from functools import reduce
import copy, time

# Try relative import first, fall back to absolute import for testing
try:
    from .support_calculation import SupportCalculator, StabilityThresholds, GeometricUtils, ThresholdManager
except ImportError:
    from support_calculation import SupportCalculator, StabilityThresholds, GeometricUtils, ThresholdManager


class Box(object):
    def __init__(self, x, y, z, lx, ly, lz):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz

    def standardize(self):
        return tuple([self.x, self.y, self.z, self.lx, self.ly, self.lz])


class Space(object):
    def __init__(self, width=10, length=10, height=10, use_enhanced_feasibility=True):
        self.plain_size = np.array([width, length, height])
        self.plain = np.zeros(shape=(width, length), dtype=np.int32)
        self.boxes = []
        self.flags = [] # record rotation information
        self.height = height
        # Initialize support calculator for enhanced feasibility checking
        self.support_calculator = SupportCalculator()
        # Initialize threshold manager for adaptive threshold adjustment
        self.threshold_manager = ThresholdManager()
        # Flag to control whether to use enhanced feasibility checking
        self.use_enhanced_feasibility = use_enhanced_feasibility
        
        # Performance monitoring and metrics collection (Requirements 4.4, 5.3, 5.4, 5.5)
        self.performance_metrics = {
            'placement_attempts': 0,
            'successful_placements': 0,
            'failed_placements': 0,
            'utilization_history': [],
            'threshold_adjustments': 0,
            'fallback_activations': 0,
            'enhanced_feasibility_usage': 0,
            'baseline_feasibility_usage': 0
        }
        
        # Performance tracking for degradation detection
        self.recent_success_rates = []
        self.recent_utilization_rates = []
        self.performance_window_size = 50  # Track last 50 operations
        self.target_utilization = 0.75  # Target >75% utilization (Requirement 5.1)
        self.baseline_utilization = 0.68  # Baseline performance to compare against
        self.degradation_threshold = 0.05  # 5% degradation triggers fallback
        
        # Fallback mechanism state
        self.fallback_active = False
        self.fallback_reason = None
        self.stable_configuration = None  # Store last known good configuration

    def print_height_graph(self):
        print(self.plain)

    def get_height_graph(self):
        plain = np.zeros(shape=self.plain_size[:2], dtype=np.int32)
        for box in self.boxes:
            plain = self.update_height_graph(plain, box)
        return plain

    @staticmethod
    def update_height_graph(plain, box):
        plain = copy.deepcopy(plain)
        le = box.lx
        ri = box.lx + box.x
        up = box.ly
        do = box.ly + box.y
        max_h = np.max(plain[le:ri, up:do])
        max_h = max(max_h, box.lz + box.z)
        plain[le:ri, up:do] = max_h
        return plain

    def get_box_list(self):
        vec = list()
        for box in self.boxes:
            vec += box.standardize()
        return vec

    def get_plain():
        return copy.deepcopy(self.plain)

    def get_action_space(self):
        return self.plain_size[0] * self.plain_size[1]

    # def get_corners(self):
    #     width = self.plain_size[0]
    #     length = self.plain_size[1]
    #     guad = [list() for _ in range(4)]

    #     guad[0].append((width, 0))
    #     guad[1].append((width, length))
    #     guad[2].append((0, length))
    #     guad[3].append((0, 0))

    #     for i in range(1, width):
    #         if self.plain[i, 0] != self.plain[i-1, 0]:
    #             guad[0].append((i, 0))
    #             guad[3].append((i, 0))

    #     for i in range(1, width):
    #         if self.plain[i, length-1] != self.plain[i-1, length-1]:
    #             guad[1].append((i, length))
    #             guad[2].append((i, length))

    #     for j in range(1, length):
    #         if self.plain[0, j] != self.plain[0, j-1]:
    #             guad[2].append((0, j))
    #             guad[3].append((0, j))

    #     for j in range(1, length):
    #         if self.plain[width-1, j] != self.plain[width-1, j]:
    #             guad[0].append((width, j))
    #             guad[1].append((width, j))

    #     for i in range(1, width):
    #         for j in range(1, length):
    #             grid_0 = self.plain[i-1, j]
    #             grid_1 = self.plain[i-1, j-1]
    #             grid_2 = self.plain[i, j-1]
    #             grid_3 = self.plain[i, j]
    #             if grid_0 == grid_1 and grid_2 == grid_3:
    #                 continue
    #             if grid_0 == grid_3 and grid_1 == grid_2:
    #                 continue
    #             if grid_0 != grid_3 or grid_0 != grid_1:
    #                 guad[0].append((i, j))
    #             if grid_1 != grid_0 or grid_1 != grid_2:
    #                 guad[1].append((i, j))
    #             if grid_2 != grid_1 or grid_2 != grid_3:
    #                 guad[2].append((i, j))
    #             if grid_3 != grid_2 or grid_3 != grid_0:
    #                 guad[3].append((i, j))

    #     return guad

    def check_box(self, plain, x, y, lx, ly, z):
        if lx+x > self.plain_size[0] or ly+y > self.plain_size[1]:
            return -1
        if lx < 0 or ly < 0:
            return -1

        rec = plain[lx:lx+x, ly:ly+y]
        r00 = rec[0,0]
        r10 = rec[x-1,0]
        r01 = rec[0,y-1]
        r11 = rec[x-1,y-1]
        rm = max(r00,r10,r01,r11)
        sc = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
        if sc < 3:
            return -1
        # get the max height
        max_h = np.max(rec)
        # check area and corner
        max_area = np.sum(rec==max_h)
        area = x * y

        # check boundary
        assert max_h >= 0
        if max_h + z > self.height:
            return -1
     
        if max_area/area > 0.95:
            return max_h
        if rm == max_h and sc == 3 and max_area/area > 0.85:
            return max_h
        if rm == max_h and sc == 4 and max_area/area > 0.50:
            return max_h

        return -1

    def get_ratio(self):
        vo = reduce(lambda x, y: x+y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * self.plain_size[2]
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    def idx_to_position(self, idx):
        lx = idx // self.plain_size[1]
        ly = idx % self.plain_size[1]
        return lx, ly

    def position_to_index(self, position):
        assert len(position) == 2
        assert position[0] >= 0 and position[1] >= 0
        assert position[0] < self.plain_size[0] and position[1] < self.plain_size[1]
        return position[0] * self.plain_size[1] + position[1]

    def drop_box(self, box_size, idx, flag):
        lx, ly = self.idx_to_position(idx)
        if not flag:
            x = box_size[0]
            y = box_size[1]
        else:
            x = box_size[1]
            y = box_size[0]
        z = box_size[2]
        plain = self.plain
        
        # Use enhanced feasibility checking if enabled
        if self.use_enhanced_feasibility:
            new_h = self.check_box_enhanced(plain, x, y, lx, ly, z)
        else:
            new_h = self.check_box(plain, x, y, lx, ly, z)
        
        # Track placement attempt and result for performance monitoring
        placement_successful = new_h != -1
        self.update_performance_metrics(placement_successful, self.use_enhanced_feasibility)
            
        if placement_successful:
            self.boxes.append(Box(x, y, z, lx, ly, new_h)) # record rotated box
            self.flags.append(flag)
            self.plain = self.update_height_graph(plain, self.boxes[-1])
            self.height = max(self.height, new_h + z)
            
            # Periodically monitor performance and adjust if needed
            if len(self.boxes) % 10 == 0:  # Check every 10 successful placements
                self.monitor_and_adjust_performance()
            
            return True
        return False

    def check_box_enhanced(self, plain, x, y, lx, ly, z):
        """
        Enhanced feasibility checking using improved stability analysis.
        
        This method implements sophisticated stability checking based on:
        1. Geometric center validation within support polygon
        2. Weighted support area calculation
        3. Corner support strength analysis
        4. Adaptive threshold management
        
        Args:
            plain: Height map of the container
            x: Width of the item
            y: Length of the item  
            lx: X position for placement
            ly: Y position for placement
            z: Height of the item
            
        Returns:
            Height at which item can be placed, or -1 if infeasible
        """
        # Basic boundary checks (same as original)
        if lx+x > self.plain_size[0] or ly+y > self.plain_size[1]:
            return -1
        if lx < 0 or ly < 0:
            return -1

        # Get height rectangle for the placement area
        rec = plain[lx:lx+x, ly:ly+y]
        max_h = np.max(rec)
        
        # Check height boundary
        if max_h + z > self.height:
            return -1
        
        # Find support points for the placement area
        support_points = self.support_calculator.find_support_points(plain, x, y, lx, ly)
        
        if not support_points:
            return -1
        
        # Compute support polygon from support points
        support_polygon = self.support_calculator.compute_support_polygon(support_points)
        
        # Get geometric center projection
        geometric_center = self.support_calculator.get_geometric_center_projection(x, y, lx, ly)
        
        # Calculate weighted support area
        support_area_ratio = self.support_calculator.calculate_weighted_support_area(
            plain, x, y, lx, ly)
        
        # Apply enhanced stability rules
        
        # Rule 1: High support area (>95%) - always approve
        if support_area_ratio > 0.95:
            return max_h
        
        # Rule 2: High support area (>70%) with geometric center validation
        if support_area_ratio > 0.70:
            if len(support_polygon.vertices) >= 3:
                if GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices):
                    return max_h
            else:
                # For degenerate polygons, use original corner logic as fallback
                r00 = rec[0,0]
                r10 = rec[x-1,0] if x > 1 else r00
                r01 = rec[0,y-1] if y > 1 else r00
                r11 = rec[x-1,y-1] if x > 1 and y > 1 else r00
                rm = max(r00,r10,r01,r11)
                sc = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
                if rm == max_h and sc >= 2:  # More relaxed
                    return max_h
        
        # Rule 3: Good support area (≥45%) with geometric center validation
        # Further relaxed to 45% to allow more feasible placements
        if support_area_ratio >= 0.45:
            if len(support_polygon.vertices) >= 3:
                if GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices):
                    return max_h
            else:
                # For degenerate polygons, use relaxed corner logic
                r00 = rec[0,0]
                r10 = rec[x-1,0] if x > 1 else r00
                r01 = rec[0,y-1] if y > 1 else r00
                r11 = rec[x-1,y-1] if x > 1 and y > 1 else r00
                rm = max(r00,r10,r01,r11)
                sc = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
                # For 45%+ support, require at least 2 corners at max height
                if rm == max_h and sc >= 2:
                    return max_h
        
        # Rule 4: Medium support area (30-44%) with relaxed corner and center validation
        if support_area_ratio > 0.30:
            # Check corner support
            r00 = rec[0,0]
            r10 = rec[x-1,0] if x > 1 else r00
            r01 = rec[0,y-1] if y > 1 else r00
            r11 = rec[x-1,y-1] if x > 1 and y > 1 else r00
            rm = max(r00,r10,r01,r11)
            sc = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
            
            # Require at least 3 corners at max height OR geometric center within support polygon
            if (rm == max_h and sc >= 3) or (len(support_polygon.vertices) >= 3 and 
                GeometricUtils.point_in_polygon(geometric_center, support_polygon.vertices)):
                return max_h
            elif len(support_polygon.vertices) < 3:
                # Fallback for degenerate cases - more lenient
                if rm == max_h and sc >= 2:
                    return max_h
        
        # Rule 5: Low support area (20-29%) with very relaxed validation for aggressive packing
        if support_area_ratio > 0.20:
            # Check corner support
            r00 = rec[0,0]
            r10 = rec[x-1,0] if x > 1 else r00
            r01 = rec[0,y-1] if y > 1 else r00
            r11 = rec[x-1,y-1] if x > 1 and y > 1 else r00
            rm = max(r00,r10,r01,r11)
            sc = int(r00==rm)+int(r10==rm)+int(r01==rm)+int(r11==rm)
            
            # Very relaxed: require at least 2 corners at max height
            if rm == max_h and sc >= 2:
                return max_h
        
        # If none of the enhanced rules pass, reject placement
        return -1

    def calculate_support_polygon(self, plain, x, y, lx, ly):
        """
        Calculate support polygon for a placement area.
        
        Args:
            plain: Height map of the container
            x: Width of the item
            y: Length of the item
            lx: X position for placement
            ly: Y position for placement
            
        Returns:
            SupportPolygon object
        """
        support_points = self.support_calculator.find_support_points(plain, x, y, lx, ly)
        return self.support_calculator.compute_support_polygon(support_points)
    
    def get_geometric_center_projection(self, x, y, lx, ly):
        """
        Get the geometric center projection for an item placement.
        
        Args:
            x: Width of the item
            y: Length of the item
            lx: X position for placement
            ly: Y position for placement
            
        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        return self.support_calculator.get_geometric_center_projection(x, y, lx, ly)
    
    def calculate_weighted_support_area(self, plain, x, y, lx, ly):
        """
        Calculate weighted support area for a placement.
        
        Args:
            plain: Height map of the container
            x: Width of the item
            y: Length of the item
            lx: X position for placement
            ly: Y position for placement
            
        Returns:
            Support area ratio (0.0 to 1.0)
        """
        return self.support_calculator.calculate_weighted_support_area(plain, x, y, lx, ly)
    
    def get_adaptive_thresholds(self, utilization_ratio):
        """
        Get adaptive stability thresholds based on container utilization.
        
        Args:
            utilization_ratio: Current space utilization (0.0 to 1.0)
            
        Returns:
            StabilityThresholds object with adjusted values
        """
        return self.threshold_manager.get_adaptive_thresholds(utilization_ratio)
    
    def adjust_thresholds_based_on_performance(self, performance_metrics):
        """
        Adjust thresholds based on current performance metrics.
        
        Args:
            performance_metrics: Dictionary containing performance data
            
        Returns:
            Updated StabilityThresholds object
        """
        return self.threshold_manager.adjust_thresholds(performance_metrics)
    
    def get_current_thresholds(self):
        """
        Get the currently active thresholds from the threshold manager.
        
        Returns:
            Current StabilityThresholds object
        """
        return self.threshold_manager.get_current_thresholds()
    
    def log_threshold_adjustment(self, metrics):
        """
        Log performance metrics for threshold adjustment analysis.
        
        Args:
            metrics: Dictionary containing performance data
        """
        self.threshold_manager.log_performance_metrics(metrics)
    
    # Performance Monitoring and Metrics Collection Methods (Task 8)
    
    def collect_utilization_metrics(self):
        """
        Collect current utilization metrics for performance monitoring.
        
        Implements Requirement 5.3: Provide utilization metrics as output
        
        Returns:
            Dictionary containing current utilization and performance metrics
        """
        current_utilization = self.get_ratio()
        
        # Calculate success rate from recent attempts
        recent_success_rate = 0.0
        if self.performance_metrics['placement_attempts'] > 0:
            recent_success_rate = (
                self.performance_metrics['successful_placements'] / 
                self.performance_metrics['placement_attempts']
            )
        
        # Calculate average recent utilization
        avg_recent_utilization = 0.0
        if self.recent_utilization_rates:
            avg_recent_utilization = sum(self.recent_utilization_rates) / len(self.recent_utilization_rates)
        
        metrics = {
            'current_utilization': current_utilization,
            'target_utilization': self.target_utilization,
            'baseline_utilization': self.baseline_utilization,
            'utilization_gap': self.target_utilization - current_utilization,
            'recent_success_rate': recent_success_rate,
            'avg_recent_utilization': avg_recent_utilization,
            'total_placements': len(self.boxes),
            'placement_attempts': self.performance_metrics['placement_attempts'],
            'successful_placements': self.performance_metrics['successful_placements'],
            'failed_placements': self.performance_metrics['failed_placements'],
            'threshold_adjustments': self.performance_metrics['threshold_adjustments'],
            'fallback_active': self.fallback_active,
            'fallback_reason': self.fallback_reason,
            'enhanced_feasibility_usage': self.performance_metrics['enhanced_feasibility_usage'],
            'baseline_feasibility_usage': self.performance_metrics['baseline_feasibility_usage']
        }
        
        return metrics
    
    def update_performance_metrics(self, placement_successful: bool, used_enhanced: bool = True):
        """
        Update performance metrics after each placement attempt.
        
        Args:
            placement_successful: Whether the placement was successful
            used_enhanced: Whether enhanced feasibility checking was used
        """
        self.performance_metrics['placement_attempts'] += 1
        
        if placement_successful:
            self.performance_metrics['successful_placements'] += 1
        else:
            self.performance_metrics['failed_placements'] += 1
        
        if used_enhanced:
            self.performance_metrics['enhanced_feasibility_usage'] += 1
        else:
            self.performance_metrics['baseline_feasibility_usage'] += 1
        
        # Update recent success rates for degradation detection
        current_success_rate = (
            self.performance_metrics['successful_placements'] / 
            self.performance_metrics['placement_attempts']
        )
        
        self.recent_success_rates.append(current_success_rate)
        if len(self.recent_success_rates) > self.performance_window_size:
            self.recent_success_rates.pop(0)
        
        # Update recent utilization rates
        current_utilization = self.get_ratio()
        self.recent_utilization_rates.append(current_utilization)
        if len(self.recent_utilization_rates) > self.performance_window_size:
            self.recent_utilization_rates.pop(0)
        
        # Store utilization history
        self.performance_metrics['utilization_history'].append(current_utilization)
    
    def detect_performance_degradation(self):
        """
        Detect if performance has degraded significantly compared to baseline.
        
        Implements Requirement 5.5: Performance degradation detection
        
        Returns:
            Tuple of (degradation_detected: bool, reason: str)
        """
        if len(self.recent_utilization_rates) < 10:  # Need sufficient data
            return False, "Insufficient data for degradation detection"
        
        avg_recent_utilization = sum(self.recent_utilization_rates) / len(self.recent_utilization_rates)
        
        # Check if utilization has dropped significantly below baseline
        utilization_degradation = self.baseline_utilization - avg_recent_utilization
        if utilization_degradation > self.degradation_threshold:
            return True, f"Utilization degraded by {utilization_degradation:.3f} below baseline {self.baseline_utilization:.3f}"
        
        # Check if success rate has dropped significantly
        if len(self.recent_success_rates) >= 10:
            avg_recent_success = sum(self.recent_success_rates) / len(self.recent_success_rates)
            if avg_recent_success < 0.3:  # Very low success rate
                return True, f"Success rate dropped to {avg_recent_success:.3f}"
        
        return False, "No significant performance degradation detected"
    
    def trigger_threshold_adjustment(self):
        """
        Trigger automatic threshold adjustment based on current performance.
        
        Implements Requirement 5.4: Automatic threshold adjustment when targets not met
        
        Returns:
            Boolean indicating whether adjustment was performed
        """
        metrics = self.collect_utilization_metrics()
        
        # Check if adjustment is needed
        utilization_gap = metrics['utilization_gap']
        recent_success_rate = metrics['recent_success_rate']
        
        adjustment_needed = (
            utilization_gap > 0.05 or  # More than 5% below target
            recent_success_rate < 0.4   # Low success rate
        )
        
        if adjustment_needed and not self.fallback_active:
            # Store current configuration as stable before adjustment
            if self.stable_configuration is None:
                self.stable_configuration = {
                    'thresholds': self.threshold_manager.get_current_thresholds(),
                    'utilization': metrics['current_utilization'],
                    'success_rate': recent_success_rate
                }
            
            # Perform threshold adjustment
            old_thresholds = self.threshold_manager.get_current_thresholds()
            new_thresholds = self.threshold_manager.adjust_thresholds(metrics)
            
            if new_thresholds != old_thresholds:
                self.performance_metrics['threshold_adjustments'] += 1
                
                # Log the adjustment (Requirement 4.4)
                adjustment_log = {
                    'timestamp': time.time(),
                    'reason': 'automatic_adjustment',
                    'utilization_gap': utilization_gap,
                    'success_rate': recent_success_rate,
                    'old_thresholds': old_thresholds,
                    'new_thresholds': new_thresholds
                }
                self.log_threshold_adjustment(adjustment_log)
                
                return True
        
        return False
    
    def activate_fallback_mechanism(self, reason: str):
        """
        Activate fallback mechanism when performance degrades significantly.
        
        Implements Requirement 5.5: Revert to previous stable configuration
        
        Args:
            reason: Reason for activating fallback
        """
        if not self.fallback_active:
            self.fallback_active = True
            self.fallback_reason = reason
            self.performance_metrics['fallback_activations'] += 1
            
            # Revert to baseline feasibility checking
            self.use_enhanced_feasibility = False
            
            # Reset thresholds to base values if we have a stable configuration
            if self.stable_configuration:
                self.threshold_manager.current_thresholds = self.stable_configuration['thresholds']
            else:
                self.threshold_manager.reset_to_base_thresholds()
            
            # Log fallback activation
            fallback_log = {
                'timestamp': time.time(),
                'reason': reason,
                'fallback_activated': True,
                'reverted_to_baseline': True
            }
            self.log_threshold_adjustment(fallback_log)
    
    def deactivate_fallback_mechanism(self):
        """
        Deactivate fallback mechanism when performance stabilizes.
        """
        if self.fallback_active:
            self.fallback_active = False
            self.fallback_reason = None
            
            # Re-enable enhanced feasibility checking
            self.use_enhanced_feasibility = True
            
            # Log fallback deactivation
            fallback_log = {
                'timestamp': time.time(),
                'reason': 'performance_stabilized',
                'fallback_activated': False,
                'enhanced_feasibility_restored': True
            }
            self.log_threshold_adjustment(fallback_log)
    
    def monitor_and_adjust_performance(self):
        """
        Comprehensive performance monitoring and adjustment method.
        
        This method should be called periodically to:
        1. Detect performance degradation
        2. Trigger threshold adjustments
        3. Activate/deactivate fallback mechanisms
        
        Returns:
            Dictionary with monitoring results and actions taken
        """
        results = {
            'degradation_detected': False,
            'threshold_adjusted': False,
            'fallback_activated': False,
            'fallback_deactivated': False,
            'metrics': self.collect_utilization_metrics()
        }
        
        # Check for performance degradation
        degradation_detected, degradation_reason = self.detect_performance_degradation()
        results['degradation_detected'] = degradation_detected
        results['degradation_reason'] = degradation_reason
        
        if degradation_detected and not self.fallback_active:
            # Activate fallback mechanism
            self.activate_fallback_mechanism(degradation_reason)
            results['fallback_activated'] = True
        elif not degradation_detected and self.fallback_active:
            # Check if we can deactivate fallback
            metrics = results['metrics']
            if (metrics['recent_success_rate'] > 0.5 and 
                metrics['current_utilization'] > self.baseline_utilization):
                self.deactivate_fallback_mechanism()
                results['fallback_deactivated'] = True
        
        # Try threshold adjustment if not in fallback mode
        if not self.fallback_active:
            adjustment_made = self.trigger_threshold_adjustment()
            results['threshold_adjusted'] = adjustment_made
        
        return results
    
    def get_performance_summary(self):
        """
        Get a comprehensive summary of performance metrics and status.
        
        Returns:
            Dictionary containing performance summary
        """
        metrics = self.collect_utilization_metrics()
        
        summary = {
            'current_status': {
                'utilization': metrics['current_utilization'],
                'target_achieved': metrics['current_utilization'] >= self.target_utilization,
                'above_baseline': metrics['current_utilization'] > self.baseline_utilization,
                'fallback_active': self.fallback_active,
                'enhanced_feasibility_enabled': self.use_enhanced_feasibility
            },
            'performance_metrics': metrics,
            'recent_performance': {
                'avg_utilization': metrics['avg_recent_utilization'],
                'success_rate': metrics['recent_success_rate'],
                'window_size': len(self.recent_utilization_rates)
            },
            'threshold_info': {
                'current_thresholds': self.threshold_manager.get_current_thresholds(),
                'adjustment_count': self.performance_metrics['threshold_adjustments'],
                'adjustment_history_length': len(self.threshold_manager.get_adjustment_history())
            },
            'system_health': {
                'degradation_detected': self.detect_performance_degradation()[0],
                'fallback_activations': self.performance_metrics['fallback_activations'],
                'stable_configuration_available': self.stable_configuration is not None
            }
        }
        
        return summary


